from openai import OpenAI
from app.rag import retrieve  # uses your retrieval code

SYSTEM_PROMPT = """შენ ხარ საქართველოს საბაჟო ინფორმაციის ასისტენტი.
შენ აუცილებლად უნდა უპასუხო ქართულად.

წესები:
1) უპასუხე მხოლოდ მოცემული "კონტექსტის" საფუძველზე.
2) თუ კონტექსტი არ არის საკმარისი, თქვი, რომ ზუსტი პასუხის დასადგენად საკმარისი ინფორმაცია არ მოიძებნა და სთხოვე მომხმარებელს დაზუსტება.
3) ყოველთვის მიუთითე წყაროები (InfoHub-ის URL-ები) ქვემოთ "წყაროები:" სექციაში.
4) არ დაამატო სხვა ბმულები; მხოლოდ infohub.rs.ge დომენი.
"""

def build_context(hits):
    # Keep context tight to control cost and reduce confusion
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] TITLE: {h['title']}\nURL: {h['url']}\nTEXT: {h['chunk']}\n"
        )
    return "\n".join(blocks)

def main():
    client = OpenAI()

    question = input("კითხვა (Georgian): ").strip()
    hits = retrieve(question, k=5)

    if not hits:
        print("\nვერ მოიძებნა შესაბამისი წყაროები.\n")
        return

    context = build_context(hits)

    user_prompt = f"""კონტექსტი (InfoHub ამონარიდები):
{context}

მომხმარებლის კითხვა:
{question}

დაბრუნების ფორმატი:
- პასუხი (ქართული, მოკლე და ზუსტი)
- საჭიროების შემთხვევაში ნაბიჯები/ჩამონათვალი
- ბოლოს: "წყაროები:" და ჩამოთვალე გამოყენებული წყაროები ფორმატით:
  [1] <URL>
  [2] <URL>
(გამოიყენე მხოლოდ ის წყაროები, რაც რეალურად გამოიყენე პასუხში.)
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    answer = resp.choices[0].message.content
    print("\n--- პასუხი ---\n")
    print(answer)

if __name__ == "__main__":
    main()
