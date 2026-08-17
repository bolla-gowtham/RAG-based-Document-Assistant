"""Query -> embed -> vector search -> (optional) MMR re-ranking for diversity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk
from src.vectorstore.store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 4,
        min_similarity: float = 0.2,
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str, use_mmr: bool = True, fetch_k: int = 12) -> list[RetrievedChunk]:
        query_emb = self.embedder.embed_query(query)
        candidates = self.store.search(query_emb, top_k=max(fetch_k, self.top_k))
        candidates = [(c, s) for c, s in candidates if s >= self.min_similarity]

        if not candidates:
            return []

        if use_mmr and len(candidates) > self.top_k:
            selected = self._mmr(query_emb, candidates, self.top_k)
        else:
            selected = candidates[: self.top_k]

        return [RetrievedChunk(chunk=c, score=s) for c, s in selected]

    def _mmr(
        self,
        query_emb: np.ndarray,
        candidates: list[tuple[Chunk, float]],
        k: int,
        lambda_mult: float = 0.5,
    ) -> list[tuple[Chunk, float]]:
        """Maximal Marginal Relevance: balances relevance to the query with
        diversity among selected chunks, so results aren't near-duplicates
        of the same passage."""
        cand_embs = self.embedder.embed_texts([c.text for c, _ in candidates])
        selected_idx: list[int] = []
        remaining_idx = list(range(len(candidates)))

        while remaining_idx and len(selected_idx) < k:
            if not selected_idx:
                # First pick: highest relevance to the query.
                scores = [candidates[i][1] for i in remaining_idx]
                best = remaining_idx[int(np.argmax(scores))]
            else:
                best, best_score = None, -1e9
                selected_embs = cand_embs[selected_idx]
                for i in remaining_idx:
                    relevance = candidates[i][1]
                    diversity_penalty = float(np.max(cand_embs[i] @ selected_embs.T))
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                    if mmr_score > best_score:
                        best, best_score = i, mmr_score
            selected_idx.append(best)
            remaining_idx.remove(best)

        return [candidates[i] for i in selected_idx]
