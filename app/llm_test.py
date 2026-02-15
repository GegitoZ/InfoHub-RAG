from openai import OpenAI

def main():
    # The client automatically reads OPENAI_API_KEY from environment
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "შენ ხარ საბაჟო ინფორმაციის ასისტენტი. ყოველთვის უპასუხე ქართულად."
            },
            {
                "role": "user",
                "content": "რა არის საბაჟო დეკლარაცია?"
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    print("\nResponse:\n")
    print(answer)


if __name__ == "__main__":
    main()
