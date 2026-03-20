import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from backend.rag_pipeline import process_query

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/ask")
async def ask_question(
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    objective: str = Form(...),
    domain: str = Form("General"),
):
    saved_paths = []

    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

            file_path = os.path.join(UPLOAD_DIR, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_paths.append(file_path)

        result = process_query(saved_paths, query, objective, domain)
        return result

    finally:
        for path in saved_paths:
            try:
                os.remove(path)
            except OSError:
                pass