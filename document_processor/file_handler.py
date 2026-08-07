import logging
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_and_chunk(file_path: str) -> list[Document]:
    """Read a PDF or .txt file and split it into overlapping chunks."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

    if not raw_text.strip():
        logger.warning(f"No extractable text found in {file_path}")
        return []

    chunks = splitter.split_text(raw_text)
    logger.info(f"Split {file_path} into {len(chunks)} chunks")
    return [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]