import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("InfoHub RAG Assistant 🇬🇪")
st.write("დასვით შეკითხვა საქართველოს საბაჟო თემებზე")

question = st.text_input("შეკითხვა:")

if st.button("კითხვა"):
    if not question:
        st.warning("გთხოვთ შეიყვანოთ შეკითხვა.")
    else:
        with st.spinner("მუშავდება..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                data = response.json()

                st.subheader("პასუხი")
                st.write(data["answer"])

                if data.get("sources"):
                    st.subheader("წყაროები")
                    for src in data["sources"]:
                        st.write(src)

            except Exception as e:
                st.error(f"შეცდომა: {e}")
