import os
from typing import Callable

from pypdf import PdfReader

LOADER_REGISTRY: dict[str, Callable[[str], str]] = {}


def register_loader(*extensions: str):
    """Register a loader function under one or more file extensions.

    A loader is any function shaped (file_path: str) -> str — it takes a
    path and returns the raw extracted text. Adding support for a new file
    format later (Ch9: tables, images, audio, video, etc.) means writing one
    new function decorated with @register_loader(...) — this dispatch logic
    itself never needs to change.
    """
    def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
        for ext in extensions:
            LOADER_REGISTRY[ext.lower()] = func
        return func
    return decorator


@register_loader(".pdf")
def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@register_loader(".txt")
def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()