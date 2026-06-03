```python
import os
import streamlit as st
import chromadb
from groq import Groq
from chromadb.utils import embedding_functions

st.set_page_config(
    page_title="DocuChat HR Assistant",
    layout="centered"
)

st.title("📄 DocuChat HR Assistant")
st.write("Ask questions about company HR policies and handbook information.")

# ---------------------------
# API Key
# ---------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

if not GROQ_API_KEY:
    st.error("Groq API key not found.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Vector Database Setup
# ---------------------------
@st.cache_resource
def init_system():

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="hr_db",
        embedding_function=emb_fn
    )

    if collection.count() == 0 and os.path.exists("Hrhandbook.txt"):

        with open("Hrhandbook.txt", "r", encoding="utf-8") as f:
            text = f.read()

        chunks = [
            chunk.strip()
            for chunk in text.split("##")
            if chunk.strip()
        ]

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks
        )

    return collection

collection = init_system()

# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# Chat Input
# ---------------------------
user_question = st.chat_input(
    "Ask a question about HR policies..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Retrieve relevant handbook chunks
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
You are an AI HR assistant helping employees understand company policies.

Use ONLY the HR handbook context provided.

Instructions:
- Answer clearly and professionally
- Explain policies in simple language
- Keep responses concise
- If information is unavailable, say:
  "I couldn't find that information in the HR handbook."

HR Handbook Context:
{context}

Employee Question:
{user_question}
"""

    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            answer = response.choices[0].message.content

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
```
