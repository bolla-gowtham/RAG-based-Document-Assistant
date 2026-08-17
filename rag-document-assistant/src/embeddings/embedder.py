"""Wraps a sentence-transformers model behind a simple, swappable interface."""
from __future__ import annotations

import numpy as np


class Embedder:
    """Thin wrapper so the rest of the codebase never imports
    sentence-transformers directly — swap models/providers here only."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded, keeps unit tests fast

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
