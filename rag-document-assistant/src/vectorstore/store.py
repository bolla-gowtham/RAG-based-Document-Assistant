"""FAISS-backed vector store with disk persistence and chunk metadata."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        import faiss  # local import keeps module import cheap for pure unit tests

        self.dim = dim
        self._faiss = faiss
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []  # positionally aligned with index vectors

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        if len(embeddings) == 0:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query_embedding = np.expand_dims(query_embedding, axis=0)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(directory / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import faiss

        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "index.faiss").exists() and (directory / "chunks.pkl").exists()
