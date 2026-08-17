"""Split documents into overlapping chunks for embedding/retrieval.

Implemented from scratch (recursive character splitting) rather than relying
on a framework, so the chunking behavior is transparent and tunable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.ingestion.loader import RawDocument

# Split on these boundaries, in order of preference, before falling back
# to a hard character cut.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split text on the first separator that produces pieces
    small enough to fit within chunk_size; falls back to a hard cut."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Hard cut as last resort.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest_separators = separators
    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    if len(parts) == 1:
        # Separator not present, try the next one.
        return _split_text(text, chunk_size, rest_separators)

    # Greedily merge parts back together up to chunk_size.
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest_separators))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])
    return overlapped


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Chunk a single RawDocument into overlapping Chunk objects."""
    separators = separators or DEFAULT_SEPARATORS
    raw_pieces = _split_text(doc.text, chunk_size, separators)
    raw_pieces = [p.strip() for p in raw_pieces if p.strip()]
    pieces = _add_overlap(raw_pieces, chunk_overlap)

    return [
        Chunk(
            id=str(uuid4()),
            text=piece,
            source=doc.source,
            chunk_index=i,
            metadata={"source": doc.source, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_documents(
    docs: list[RawDocument], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
