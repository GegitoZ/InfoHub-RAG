import re
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from app.rag import retrieve

app = FastAPI(title="InfoHub RAG API")

client = OpenAI()

SYSTEM_PROMPT = """შენ ხარ საქართველოს საბაჟო ინფორმაციის ასისტენტი.
აუცილებლად უპასუხე ქართულად.

წესები:
1) უპასუხე მხოლოდ მოცემული კონტექსტის საფუძველზე (InfoHub ამონარიდები).
2) თუ კონტექსტი არ არის საკმარისი ზუსტი პასუხისთვის, თქვი რომ ინფორმაცია არ არის საკმარისი და სთხოვე მომხმარებელს დაზუსტება.
3) პასუხში აუცილებლად გამოიყენე ციტირებები კვადრატულ ფრჩხილებში, მაგალითად [1], [2] და ა.შ.
4) ციტირებების ნომრები უნდა შეესაბამებოდეს ქვემოთ მოცემულ კონტექსტში არსებულ ნომრებს.
"""


class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "გთხოვთ შეიყვანოთ კითხვა.", "sources": []}

    # Retrieve top-k chunks
    hits = retrieve(question, k=5)

    if not hits:
        return {"answer": "შესაბამისი ინფორმაცია ვერ მოიძებნა.", "sources": []}

    # Build numbered context and a map: number -> URL
    context_parts = []
    sources_map = {}  # {1: url1, 2: url2, ...}

    for i, h in enumerate(hits, start=1):
        context_parts.append(f"[{i}] {h['chunk']}")
        sources_map[i] = h["url"]

    context = "\n\n".join(context_parts)

    user_prompt = f"""კონტექსტი (InfoHub ამონარიდები):
{context}

მომხმარებლის კითხვა:
{question}

დაბრუნების ფორმატი:
- პასუხი ქართულად
- საჭიროების შემთხვევაში ჩამონათვალი/ნაბიჯები
- პასუხში ჩასვი ციტირებები, მაგალითად [1], [2]
(არ დაამატო სხვა წყაროები; გამოიყენე მხოლოდ მოცემული კონტექსტის ნომრები.)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    answer = response.choices[0].message.content or ""

    # Extract citations like [1], [2] from the answer
    cited_nums = re.findall(r"\[(\d+)\]", answer)
    cited_nums = [int(n) for n in cited_nums if n.isdigit()]
    cited_nums = [n for n in cited_nums if n in sources_map]

    # Deduplicate while keeping order
    seen = set()
    ordered = []
    for n in cited_nums:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    used_sources = [sources_map[n] for n in ordered]

    # Fallback: if model forgot citations, return top 2 retrieved sources
    if not used_sources:
        used_sources = list(sources_map.values())[:2]

    return {
        "answer": answer,
        "sources": used_sources
    }
