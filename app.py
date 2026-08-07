import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from agents.workflow import AgentWorkflow
from document_processor.file_handler import load_and_chunk
from retriever.builder import RetrieverBuilder
from utils.logging import logger

app = FastAPI(title="Trusty (Chapter 5 — logging, validation, error handling)")
workflow = AgentWorkflow()
retriever_builder = RetrieverBuilder()

state = {"retriever": None}

UPLOAD_DIR = ".cache/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
def upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_and_chunk(dest_path)
    if not docs:
        raise HTTPException(status_code=422, detail="No extractable text found in this file.")

    state["retriever"] = retriever_builder.build_hybrid_retriever(docs)
    logger.info(f"Indexed {file.filename}: {len(docs)} chunks")
    return {"status": "indexed", "chunks": len(docs)}


class Question(BaseModel):
    text: str


@app.post("/ask")
def ask(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if state["retriever"] is None:
        raise HTTPException(status_code=400, detail="No document indexed yet. Call /upload first.")

    try:
        documents = state["retriever"].invoke(q.text)
        result = workflow.full_pipeline(q.text, documents)
        logger.info(f"Answered question: {q.text[:60]!r}")
        return result
    except Exception as e:
        logger.error(f"Pipeline failed for question {q.text!r}: {e}")
        raise HTTPException(status_code=500, detail=f"Something went wrong processing this question: {e}")