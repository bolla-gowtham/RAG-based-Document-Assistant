"""Load raw documents from disk into plain text with source metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class RawDocument:
    text: str
    source: str  # file path, used for citations


def _load_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str | Path) -> RawDocument:
    """Load a single supported file into a RawDocument."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if path.suffix.lower() == ".pdf":
        text = _load_pdf(path)
    else:
        text = _load_txt_or_md(path)

    text = text.strip()
    if not text:
        raise ValueError(f"No extractable text found in {path}")

    return RawDocument(text=text, source=str(path))


def load_directory(directory: str | Path) -> list[RawDocument]:
    """Recursively load every supported file under a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: list[RawDocument] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs.append(load_document(fpath))
                except ValueError as e:
                    print(f"[loader] skipping {fpath}: {e}")
    return docs
