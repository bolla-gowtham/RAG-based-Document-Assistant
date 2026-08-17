"""Provider-agnostic answer generation grounded in retrieved chunks.

Swap OpenAI <-> Anthropic without touching any other module: both
implementations satisfy the same `generate(prompt) -> str` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import settings
from src.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the provided context. Every claim must be traceable to a numbered source. "
    "Cite sources inline like [1], [2]. If the context does not contain enough "
    "information to answer confidently, say 'I don't have enough information in "
    "the provided documents to answer that.' Do not use outside knowledge."
)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


def build_llm_client(provider: str | None = None) -> LLMClient:
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        return OpenAIClient(settings.openai_api_key, settings.openai_model)
    if provider == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    raise ValueError(f"Unknown LLM provider: {provider}")


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {rc.chunk.source})\n{rc.chunk.text}")
    return "\n\n".join(lines)


def build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"id": i, "source": rc.chunk.source, "score": round(rc.score, 4), "text": rc.chunk.text}
        for i, rc in enumerate(chunks, start=1)
    ]


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or build_llm_client()

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> dict:
        if not retrieved:
            return {
                "answer": "I don't have enough information in the provided documents to answer that.",
                "sources": [],
            }

        context = build_context_block(retrieved)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely, citing sources like [1], [2] inline."
        )
        answer_text = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)
        return {"answer": answer_text, "sources": build_sources(retrieved)}
