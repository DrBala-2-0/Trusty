import os
import shutil
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from agents.workflow import AgentWorkflow
from document_processor.file_handler import load_and_chunk
from retriever.builder import RetrieverBuilder

app = FastAPI(title="Trusty (Chapter 4 — real retrieval)")
workflow = AgentWorkflow()
retriever_builder = RetrieverBuilder()

# In-memory for now — one retriever per process, reset on restart.
# Persistence/multi-doc-session handling comes in a later chapter.
state = {"retriever": None}

UPLOAD_DIR = ".cache/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
def upload(file: UploadFile):
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_and_chunk(dest_path)
    if not docs:
        return {"error": "No extractable text found in this file."}

    state["retriever"] = retriever_builder.build_hybrid_retriever(docs)
    return {"status": "indexed", "chunks": len(docs)}


class Question(BaseModel):
    text: str

@app.post("/ask")
def ask(q: Question):
    if state["retriever"] is None:
        return {"error": "No document indexed yet. Call /upload first."}

    documents = state["retriever"].invoke(q.text)
    result = workflow.full_pipeline(q.text, documents)
    return result