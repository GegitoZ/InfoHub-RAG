import re
import streamlit as st
from openai import OpenAI
from app.rag import retrieve

# Streamlit page setup
st.set_page_config(page_title="InfoHub RAG Assistant", page_icon="🇬🇪")
st.title("InfoHub RAG Assistant 🇬🇪")
st.write("დასვით შეკითხვა საქართველოს საბაჟო თემებზე")

SYSTEM_PROMPT = """შენ ხარ საქართველოს საბაჟო ინფორმაციის ასისტენტი.
აუცილებლად უპასუხე ქართულად.

წესები:
1) გამოიყენე მხოლოდ მოცემული კონტექსტი.
2) თუ ინფორმაცია საკმარისი არ არის, თქვი რომ მონაცემები არ არის საკმარისი.
3) პასუხში გამოიყენე ციტირებები [1], [2] და ა.შ.
4) უპასუხე მოკლედ და კონკრეტულად.
"""


def extract_used_sources(answer, sources_map):
    numbers = re.findall(r"\[(\d+)\]", answer)
    numbers = [int(n) for n in numbers if int(n) in sources_map]

    seen = set()
    used = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            used.append(sources_map[n])

    if not used:
        used = list(sources_map.values())[:2]

    return used


question = st.text_input("შეკითხვა:")

if st.button("კითხვა"):
    if not question.strip():
        st.warning("გთხოვთ შეიყვანოთ შეკითხვა")
        st.stop()

    with st.spinner("ვიძიებ ინფორმაციას..."):
        hits = retrieve(question, k=5)

    if not hits:
        st.error("შესაბამისი ინფორმაცია ვერ მოიძებნა")
        st.stop()

    context = ""
    sources_map = {}

    for i, h in enumerate(hits, start=1):
        context += f"[{i}] {h['chunk']}\n\n"
        sources_map[i] = h["url"]

    prompt = f"""
კონტექსტი:
{context}

მომხმარებლის კითხვა:
{question}
"""

    client = OpenAI()

    with st.spinner("მუშავდება"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=350,
        )

    answer = response.choices[0].message.content

    st.subheader("პასუხი")
    st.write(answer)

    st.subheader("წყაროები")
    sources = extract_used_sources(answer, sources_map)
    for s in sources:
        st.write(s)
