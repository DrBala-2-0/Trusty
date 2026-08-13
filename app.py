import os
import shutil
from typing import Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from agents.workflow import AgentWorkflow
from document_processor.file_handler import chunk_text, load_and_chunk
from document_processor.url_loader import fetch_url_text
from retriever.builder import RetrieverBuilder
from utils.logging import logger
from utils.session import resolve_session_id

app = FastAPI(title="Trusty (Chapter 7 — multi-user retrieval)")
workflow = AgentWorkflow()
retriever_builder = RetrieverBuilder()

# Session-scoped state. Both are plain in-memory dicts — wiped on process
# restart (including uvicorn --reload). That's a deliberate choice, not an
# oversight: see docs/chapters/chapter-7.md for why on-disk session
# persistence isn't warranted by anything the blueprint actually requires.
session_docs: dict[str, list] = {}   # session_id -> every chunk uploaded so far, across all /upload calls
retrievers: dict[str, object] = {}   # session_id -> that session's current hybrid retriever

UPLOAD_DIR = ".cache/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
def upload(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: str = Depends(resolve_session_id),
):
    file_given = file is not None and bool(file.filename)
    if file_given and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a url, not both.")
    if not file_given and not url:
        raise HTTPException(status_code=400, detail="No file or url provided.")

    if url:
        try:
            raw_text = fetch_url_text(url)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        new_docs = chunk_text(raw_text, source=url)
        skipped = []
        label = url
    else:
        dest_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            new_docs, skipped = load_and_chunk(dest_path)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        label = file.filename

    if not new_docs:
        if skipped:
            raise HTTPException(
                status_code=422,
                detail=f"No extractable content found. All files were skipped: {skipped}",
            )
        raise HTTPException(status_code=422, detail="No extractable text found.")

    session_docs.setdefault(session_id, []).extend(new_docs)
    all_docs = session_docs[session_id]

    retrievers[session_id] = retriever_builder.build_hybrid_retriever(all_docs, session_id)
    logger.info(
        f"[{session_id}] Indexed {label}: +{len(new_docs)} chunks "
        f"({len(all_docs)} total for this session)"
    )
    return {
        "status": "indexed",
        "chunks_added": len(new_docs),
        "chunks_total": len(all_docs),
        "session_id": session_id,
        "skipped": skipped,
    }



class Question(BaseModel):
    text: str


@app.post("/ask")
def ask(q: Question, session_id: str = Depends(resolve_session_id)):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if session_id not in retrievers:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet for this session. Call /upload first.",
        )

    try:
        documents = retrievers[session_id].invoke(q.text)
        result = workflow.full_pipeline(q.text, documents)
        logger.info(f"[{session_id}] Answered question: {q.text[:60]!r}")
        return result
    except Exception as e:
        logger.error(f"[{session_id}] Pipeline failed for question {q.text!r}: {e}")
        raise HTTPException(status_code=500, detail=f"Something went wrong processing this question: {e}")