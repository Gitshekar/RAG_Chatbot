import os
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from backend.embeddings import HashingEmbeddings
from backend.metrics import (
    calculate_answer_accuracy,
    calculate_faithfulness,
    calculate_relevance,
    is_refusal,
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm(objective: str):
    # Objective 1 is intentionally commented out because it uses a local Ollama model.
    # if objective == "Objective 1: llama3.1":
    #     return OllamaLLM(model="llama3.1")

    if objective == "Objective 2: llama-3.1-8b-instant":
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
        )

    if objective == "Objective 3: openai/gpt-oss-20b":
        return ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
        )

    raise ValueError("Invalid objective")


def load_and_split_pdfs(file_paths):
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


def run_rag(llm, chunks, query, domain):
    embeddings = HashingEmbeddings(dim=256)

    with tempfile.TemporaryDirectory(prefix="chroma_") as persist_dir:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
You are a DOCUMENT-BOUND EXPLANATORY ASSISTANT.

TASK:
Produce a detailed explanation using ONLY the information provided.

RULES:
1. Use ONLY the provided information.
2. Write at least 5-7 sentences when possible.
3. Elaborate only on explicitly stated points.
4. Do not introduce new concepts.
5. If the answer is not derivable, reply exactly:
"I cannot answer this question based on the provided documents."

DOMAIN: {domain}

CONTENT:
{{context}}
"""),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        response = rag_chain.invoke({"input": query})

        docs = response["context"]
        answer = response["answer"]

        acc = calculate_answer_accuracy(answer, docs)
        faith = calculate_faithfulness(answer, docs)
        rel = calculate_relevance(query, answer)

        if is_refusal(answer):
            acc = faith = rel = 0.0

        return answer, acc, faith, rel


def process_query(file_paths, query, objective, domain):
    chunks = load_and_split_pdfs(file_paths)
    llm = get_llm(objective)

    answer, acc, faith, rel = run_rag(llm, chunks, query, domain)

    return {
        "answer": answer,
        "accuracy": acc,
        "faithfulness": faith,
        "relevance": rel,
    }