import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processor.loaders import LOADER_REGISTRY, SHEET_LOADER_REGISTRY
from utils.logging import logger

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Extensions whose raw text starts with a single header line that should be
# repeated on every chunk. .xlsx is handled separately (SHEET_LOADER_REGISTRY
# below) since it can have multiple sheets, each needing its own header.
HEADER_REPEAT_EXTENSIONS = {".csv"}


def chunk_text(raw_text: str, source: str, extra_metadata: dict | None = None, header: str | None = None) -> list[Document]:
    """Split raw text into overlapping Document chunks tagged with source metadata.

    If header is given, it's prepended to every resulting chunk. Shared by
    every ingestion path, not just the extension-keyed registries in
    load_and_chunk() — zip and URL ingestion (Ch9) call this directly.
    """
    if not raw_text.strip():
        logger.warning(f"No extractable text found for {source}")
        return []

    metadata = {"source": source}
    if extra_metadata:
        metadata.update(extra_metadata)

    chunks = splitter.split_text(raw_text)
    logger.info(f"Split {source} into {len(chunks)} chunks")

    docs = []
    for chunk in chunks:
        content = f"{header}\n{chunk}" if header else chunk
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_and_chunk(file_path: str) -> list[Document]:
    """Look up a loader for file_path's extension, extract raw text, and
    chunk it via chunk_text().

    Raises ValueError if no loader is registered for the file's extension —
    a distinct failure from "loader ran but found no text" (empty PDF page,
    blank .txt, empty sheet), which still returns [] as before.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in SHEET_LOADER_REGISTRY:
        sheets = SHEET_LOADER_REGISTRY[ext](file_path)
        docs = []
        for sheet_name, header_line, body_text in sheets:
            docs.extend(
                chunk_text(
                    body_text,
                    source=f"{file_path} [sheet: {sheet_name}]",
                    header=f"Sheet: {sheet_name}\n{header_line}",
                )
            )
        return docs

    loader = LOADER_REGISTRY.get(ext)
    if loader is None:
        supported = ", ".join(sorted(set(LOADER_REGISTRY) | set(SHEET_LOADER_REGISTRY))) or "(none registered)"
        raise ValueError(
            f"Unsupported file type '{ext or '(no extension)'}'. "
            f"Supported types: {supported}"
        )

    raw_text = loader(file_path)

    if ext in HEADER_REPEAT_EXTENSIONS:
        first_line, _, rest = raw_text.partition("\n")
        return chunk_text(rest, source=file_path, header=first_line)

    return chunk_text(raw_text, source=file_path)