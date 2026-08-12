import os
import shutil
import tempfile
import zipfile

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processor.loaders import LOADER_REGISTRY, SHEET_LOADER_REGISTRY
from utils.logging import logger

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

HEADER_REPEAT_EXTENSIONS = {".csv"}

# Reject a zip whose declared (pre-extraction) total size exceeds this —
# a cheap guard against decompression bombs (a tiny zip that expands to
# gigabytes). Checked before anything is extracted.
MAX_ZIP_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # 200 MB


def chunk_text(raw_text: str, source: str, extra_metadata: dict | None = None, header: str | None = None) -> list[Document]:
    """Split raw text into overlapping Document chunks tagged with source metadata.

    If header is given, it's prepended to every resulting chunk. Shared by
    every ingestion path, not just the extension-keyed registries in
    load_and_chunk() — zip (below) and URL ingestion (Ch9 Step 6) call this
    directly.
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


def load_and_chunk(file_path: str) -> tuple[list[Document], list[dict]]:
    """Extract and chunk file_path, dispatching by extension.

    Returns (docs, skipped) — skipped is always [] except when file_path is
    a .zip whose contents included files with unrecognized extensions (see
    _load_zip below). Raises ValueError for a directly-uploaded unsupported
    file (unchanged from Ch8/Ch9 Step 1-4), or RuntimeError if a registered
    loader's processing itself fails (Ch9 Step 3/4's vision/audio errors).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".zip":
        return _load_zip(file_path)

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
        return docs, []

    loader = LOADER_REGISTRY.get(ext)
    if loader is None:
        supported = ", ".join(sorted(set(LOADER_REGISTRY) | set(SHEET_LOADER_REGISTRY) | {".zip"})) or "(none registered)"
        raise ValueError(
            f"Unsupported file type '{ext or '(no extension)'}'. "
            f"Supported types: {supported}"
        )

    raw_text = loader(file_path)

    if ext in HEADER_REPEAT_EXTENSIONS:
        first_line, _, rest = raw_text.partition("\n")
        return chunk_text(rest, source=file_path, header=first_line), []

    return chunk_text(raw_text, source=file_path), []


def _load_zip(file_path: str) -> tuple[list[Document], list[dict]]:
    """Extract a zip's entries and recurse load_and_chunk() on each.

    Unsupported-format entries are skipped and reported (not aborted) —
    real-world zips routinely contain junk (.DS_Store, __MACOSX/, etc.).
    A genuine processing failure on a recognized format (RuntimeError, e.g.
    a vision/audio API error) is NOT caught here — it aborts the whole
    archive upload, matching Ch9 Step 3/4's fail-loud precedent: unsupported
    formats are a scope gap, a real processing failure is not.

    Two guards before any extraction happens: a size cap (decompression-bomb
    protection) and per-entry path validation (zip-slip protection) — an
    entry like '../../etc/passwd' would otherwise write outside the temp
    directory."""
    display_name = os.path.basename(file_path)
    docs: list[Document] = []
    skipped: list[dict] = []

    with zipfile.ZipFile(file_path) as zf:
        infolist = zf.infolist()

        total_uncompressed = sum(info.file_size for info in infolist)
        if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"Archive '{display_name}' too large when extracted "
                f"({total_uncompressed / 1_048_576:.1f} MB > "
                f"{MAX_ZIP_UNCOMPRESSED_BYTES / 1_048_576:.0f} MB limit)."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = os.path.abspath(tmp_dir)

            for info in infolist:
                if info.is_dir():
                    continue

                inner_name = info.filename
                target_path = os.path.abspath(os.path.join(tmp_root, inner_name))

                if not (target_path == tmp_root or target_path.startswith(tmp_root + os.sep)):
                    raise ValueError(
                        f"Archive '{display_name}' contains an unsafe path "
                        f"('{inner_name}') and was rejected."
                    )

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                try:
                    inner_docs, inner_skipped = load_and_chunk(target_path)
                except ValueError as e:
                    skipped.append({"file": f"{display_name} -> {inner_name}", "reason": str(e)})
                    continue

                for doc in inner_docs:
                    doc.metadata["source"] = f"{display_name} -> {inner_name}"
                docs.extend(inner_docs)

                for s in inner_skipped:
                    skipped.append({"file": f"{display_name} -> {s['file']}", "reason": s["reason"]})

    return docs, skipped