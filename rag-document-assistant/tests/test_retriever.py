import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore


class FakeEmbedder:
    """Deterministic fake embedder so tests don't download a real model."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._vocab = {}

    def _vec_for(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding based on hash, normalized.
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        v = rng.rand(self.dim).astype("float32")
        return v / np.linalg.norm(v)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec_for(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec_for(text)


@pytest.fixture
def populated_store():
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(dim=8)
    chunks = [
        Chunk(id="1", text="The cat sat on the mat.", source="a.txt", chunk_index=0),
        Chunk(id="2", text="Dogs are loyal animals.", source="b.txt", chunk_index=0),
        Chunk(id="3", text="Cats are independent pets.", source="a.txt", chunk_index=1),
    ]
    embeddings = embedder.embed_texts([c.text for c in chunks])
    store.add(embeddings, chunks)
    return embedder, store, chunks


def test_vectorstore_add_and_search_returns_results(populated_store):
    embedder, store, chunks = populated_store
    query_emb = embedder.embed_query("cats")
    results = store.search(query_emb, top_k=2)
    assert len(results) == 2
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert -1.0 <= score <= 1.0


def test_vectorstore_empty_search_returns_empty_list():
    store = VectorStore(dim=8)
    results = store.search(np.zeros(8, dtype="float32"), top_k=3)
    assert results == []


def test_vectorstore_mismatched_lengths_raises():
    store = VectorStore(dim=8)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 8), dtype="float32"), [Chunk(id="1", text="x", source="a", chunk_index=0)])


def test_retriever_respects_top_k(populated_store):
    embedder, store, chunks = populated_store
    retriever = Retriever(embedder, store, top_k=2, min_similarity=-1.0)
    results = retriever.retrieve("cats and dogs", use_mmr=False)
    assert len(results) == 2


def test_retriever_filters_by_min_similarity(populated_store):
    embedder, store, chunks = populated_store
    retriever = Retriever(embedder, store, top_k=3, min_similarity=1.1)  # impossible threshold
    results = retriever.retrieve("cats", use_mmr=False)
    assert results == []


def test_retriever_mmr_returns_requested_count(populated_store):
    embedder, store, chunks = populated_store
    retriever = Retriever(embedder, store, top_k=2, min_similarity=-1.0)
    results = retriever.retrieve("pets", use_mmr=True, fetch_k=3)
    assert len(results) == 2
