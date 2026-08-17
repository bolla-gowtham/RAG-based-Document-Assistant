"""Orchestrates the end-to-end RAG flow: ingest documents, or answer a query."""
from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.embeddings.embedder import Embedder
from src.generation.generator import AnswerGenerator
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_directory, load_document
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder(settings.embedding_model)
        self.store: VectorStore | None = None
        self._generator: AnswerGenerator | None = None

    # ---------- Ingestion ----------
    def ingest_directory(self, directory: str) -> int:
        docs = load_directory(directory)
        return self._ingest_docs(docs)

    def ingest_file(self, path: str) -> int:
        doc = load_document(path)
        return self._ingest_docs([doc])

    def _ingest_docs(self, docs) -> int:
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_texts([c.text for c in chunks])

        if self.store is None:
            if VectorStore.exists(settings.vector_db_dir):
                self.store = VectorStore.load(settings.vector_db_dir)
            else:
                self.store = VectorStore(dim=self.embedder.dim)

        self.store.add(embeddings, chunks)
        self.store.save(settings.vector_db_dir)
        return len(chunks)

    # ---------- Query ----------
    def _ensure_store_loaded(self):
        if self.store is None:
            if not VectorStore.exists(settings.vector_db_dir):
                raise RuntimeError(
                    "No vector store found. Run `python scripts/ingest.py --source <dir>` first."
                )
            self.store = VectorStore.load(settings.vector_db_dir)

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator()
        return self._generator

    def query(self, question: str) -> dict:
        self._ensure_store_loaded()
        retriever = Retriever(
            self.embedder, self.store, top_k=settings.top_k, min_similarity=settings.min_similarity
        )
        retrieved = retriever.retrieve(question)
        return self.generator.answer(question, retrieved)

    def retrieve_only(self, question: str, top_k: int | None = None):
        """Useful for evaluation: skip the LLM call, just check retrieval quality."""
        self._ensure_store_loaded()
        retriever = Retriever(
            self.embedder,
            self.store,
            top_k=top_k or settings.top_k,
            min_similarity=settings.min_similarity,
        )
        return retriever.retrieve(question)
