import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processor.loaders import LOADER_REGISTRY
from utils.logging import logger

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_and_chunk(file_path: str) -> list[Document]:
    """Extract text from a file (via the registered loader for its extension)
    and split it into overlapping chunks.

    Raises ValueError if no loader is registered for the file's extension —
    a distinct failure from "loader ran but found no text" (empty PDF page,
    blank .txt), which still returns [] as before."""
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADER_REGISTRY.get(ext)
    if loader is None:
        supported = ", ".join(sorted(LOADER_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unsupported file type '{ext or '(no extension)'}'. "
            f"Supported types: {supported}"
        )

    raw_text = loader(file_path)

    if not raw_text.strip():
        logger.warning(f"No extractable text found in {file_path}")
        return []

    chunks = splitter.split_text(raw_text)
    logger.info(f"Split {file_path} into {len(chunks)} chunks")
    return [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]