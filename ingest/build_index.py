import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw_docs")
OUT_INDEX = Path("data/index.faiss")
OUT_META = Path("data/index_meta.json")

MODEL_NAME = "intfloat/multilingual-e5-base"  # multilingual, good for Georgian

# --- Cleaning helpers ---

WS_RE = re.compile(r"\s+")
NON_WORDY_RE = re.compile(r"^[\W\d_]+$")

# A tiny list of common UI-ish phrases you may see; safe to drop if a line matches
UI_PHRASES = [
    "Sign in", "Log in", "Login", "Register", "Search", "Menu",
    "Cookies", "Privacy", "Terms", "Language",
    # Georgian variants (not exhaustive, just helpful)
    "შესვლა", "რეგისტრაცია", "ძიება", "მენიუ", "ქუქი", "კონფიდენციალურობა"
]

def normalize_line(line: str) -> str:
    line = WS_RE.sub(" ", line).strip()
    return line

def looks_like_ui(line: str) -> bool:
    if len(line) < 3:
        return True
    if NON_WORDY_RE.match(line):
        return True
    # drop if line is exactly/mostly UI phrase
    low = line.lower()
    for p in UI_PHRASES:
        if low == p.lower():
            return True
    return False

def clean_text_to_lines(text: str):
    # Start from innerText-style content; split to lines to remove repeated junk
    raw_lines = text.splitlines()
    lines = []
    for ln in raw_lines:
        ln = normalize_line(ln)
        if not ln:
            continue
        if looks_like_ui(ln):
            continue
        lines.append(ln)
    return lines

def drop_globally_common_lines(all_docs_lines, common_threshold=0.35):
    """
    Remove lines that appear in too many documents (nav/footer repeated everywhere).
    common_threshold = fraction of docs that contain the line.
    """
    doc_count = len(all_docs_lines)
    presence = Counter()
    for lines in all_docs_lines:
        for ln in set(lines):
            presence[ln] += 1

    too_common = {ln for ln, c in presence.items() if c / doc_count >= common_threshold}
    filtered = []
    for lines in all_docs_lines:
        filtered.append([ln for ln in lines if ln not in too_common])
    return filtered

def chunk_text(text: str, chunk_size=900, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# --- Main build ---

def main():
    files = sorted(RAW_DIR.glob("doc_*.json"))
    if not files:
        raise SystemExit("No raw docs found in data/raw_docs. Run fetch_docs.py first.")

    docs = []
    all_lines = []
    for fp in files:
        doc = json.loads(fp.read_text(encoding="utf-8"))
        lines = clean_text_to_lines(doc.get("text", ""))
        all_lines.append(lines)
        docs.append({
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "lines": lines
        })

    # Drop lines that are repeated across many docs (footer/nav)
    all_lines = drop_globally_common_lines(all_lines, common_threshold=0.35)
    for d, lines in zip(docs, all_lines):
        d["lines"] = lines

    # Join back to text for chunking
    metas = []
    passages = []

    for d in docs:
        url, title = d["url"], d["title"]
        text = " ".join(d["lines"])
        text = WS_RE.sub(" ", text).strip()

        # Skip docs that became too short after cleaning
        if len(text) < 1200:
            continue

        chunks = chunk_text(text, chunk_size=900, overlap=200)

        for ch in chunks:
            # Skip tiny chunks
            if len(ch) < 300:
                continue
            metas.append({"url": url, "title": title, "chunk": ch})
            passages.append("passage: " + ch)

    if not metas:
        raise SystemExit("After cleaning, no chunks left. Lower thresholds or check raw text.")

    print(f"Chunks to embed: {len(metas)}")

    embedder = SentenceTransformer(MODEL_NAME)
    vecs = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype="float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(vecs)

    faiss.write_index(index, str(OUT_INDEX))
    OUT_META.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Index built successfully")
    print(f"Saved: {OUT_INDEX}")
    print(f"Saved: {OUT_META}")
    print(f"Total chunks indexed: {len(metas)}")

if __name__ == "__main__":
    main()
