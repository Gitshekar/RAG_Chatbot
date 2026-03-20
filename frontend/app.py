import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Statistics RAG Chatbot", layout="wide")
st.title("📊 Conversational RAG Chatbot For Statistical Learning PDFs")

with st.sidebar:
    st.header("🎯 Select Objective")
    objective = st.radio(
        "Choose Objective",
        [
            # "Objective 1: llama3.1",  # local Ollama model, intentionally disabled
            "Objective 2: llama-3.1-8b-instant",
            "Objective 3: openai/gpt-oss-20b",
        ],
    )

st.subheader("📂 Upload PDFs")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
)

query = st.text_input("Ask a question from PDFs")
domain = st.text_input("Domain", value="General")

if st.button("Get Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    files = [
        ("files", (f.name, f.getvalue(), "application/pdf"))
        for f in uploaded_files
    ]

    data = {
        "query": query,
        "objective": objective,
        "domain": domain,
    }

    with st.spinner("Analyzing PDF and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                files=files,
                data=data,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            st.success("Answer generated successfully!")
            st.write(result["answer"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f'{result["accuracy"]}%')
            col2.metric("Faithfulness", f'{result["faithfulness"]}%')
            col3.metric("Relevance", f'{result["relevance"]}%')

        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")