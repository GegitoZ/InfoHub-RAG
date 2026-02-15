import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL_NAME = "intfloat/multilingual-e5-base"

_index = None
_meta = None
_embedder = None
_bm25 = None
_bm25_tokens = None


def load_store():
    global _index, _meta, _embedder, _bm25, _bm25_tokens

    if _meta is None:
        _meta = json.load(open("data/index_meta.json", "r", encoding="utf-8"))

    # BM25 is built only from meta chunks; doesn't need FAISS at all
    if _bm25 is None:
        _bm25_tokens = [_tokenize(m["chunk"]) for m in _meta]
        _bm25 = BM25Okapi(_bm25_tokens)

    # Semantic components (optional rerank)
    if _index is None:
        _index = faiss.read_index("data/index.faiss")
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)


def _tokenize(s: str):
    # keeps Georgian letters (ა-ჰ), latin letters, digits
    s = s.lower()
    s = re.sub(r"[^\w\sა-ჰ]", " ", s)
    return [t for t in s.split() if len(t) >= 2]


def retrieve(question: str, k: int = 5, bm25_candidates: int = 60, use_semantic_rerank: bool = True):
    """
    BM25-first retrieval:
    1) Rank ALL chunks by BM25 (keyword relevance)
    2) Take top N candidates (e.g., 60)
    3) (Optional) rerank those candidates by semantic similarity
    """
    load_store()

    q_tokens = _tokenize(question)
    bm25_scores = _bm25.get_scores(q_tokens)

    # Take top bm25 indices
    top_idx = np.argsort(bm25_scores)[::-1][:bm25_candidates]

    candidates = []
    for idx in top_idx:
        m = _meta[idx]
        candidates.append({
            "idx": int(idx),
            "bm25": float(bm25_scores[idx]),
            "title": m["title"],
            "url": m["url"],
            "chunk": m["chunk"],
        })

    if use_semantic_rerank:
        # compute semantic similarity only for these candidates
        qvec = _embedder.encode(["query: " + question], normalize_embeddings=True)
        qvec = np.asarray(qvec, dtype="float32")

        # We need vectors for these indices; FAISS IndexFlatIP doesn't support direct fetch,
        # so we approximate by searching FAISS widely and mapping scores.
        # Simple trick: build a dict of semantic scores for many nearest neighbors.
        sem_k = max(400, bm25_candidates * 5)
        sem_scores, sem_idxs = _index.search(qvec, sem_k)
        sem_map = {int(i): float(s) for s, i in zip(sem_scores[0], sem_idxs[0])}

        # Normalize BM25 for mixing
        bm_vals = [c["bm25"] for c in candidates]
        bm_min, bm_max = min(bm_vals), max(bm_vals)
        denom = (bm_max - bm_min) if (bm_max - bm_min) > 1e-9 else 1.0

        for c in candidates:
            bm_norm = (c["bm25"] - bm_min) / denom
            sem = sem_map.get(c["idx"], 0.0)  # 0 if not in sem neighbors
            c["semantic"] = sem
            # combine: keyword heavy + semantic support
            c["combined"] = 0.75 * bm_norm + 0.25 * max(0.0, sem)
    else:
        for c in candidates:
            c["semantic"] = 0.0
            c["combined"] = c["bm25"]

    candidates.sort(key=lambda x: x["combined"], reverse=True)

    # de-dupe by URL so results cover multiple docs
    final = []
    seen = set()
    for c in candidates:
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        final.append(c)
        if len(final) >= k:
            break

    return final


if __name__ == "__main__":
    print("BM25-first Retrieval Test")
    q = input("კითხვა (Georgian): ").strip()
    hits = retrieve(q, k=5, bm25_candidates=80, use_semantic_rerank=True)

    print("\nTop matches:\n")
    for i, h in enumerate(hits, start=1):
        print(f"{i}. combined={h['combined']:.3f}  bm25={h['bm25']:.3f}  sem={h['semantic']:.3f}")
        print(f"   title: {h['title']}")
        print(f"   url:   {h['url']}")
        print(f"   chunk: {h['chunk'][:320]}...")
        print()
