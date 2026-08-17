"""Evaluation harness for the RAG pipeline.

Two kinds of metrics:
1. Retrieval quality (reference-free): hit-rate@k and mean reciprocal rank,
   computed against a labeled QA set where each question names the source
   file that should be retrieved.
2. Generation quality (LLM-as-judge): faithfulness (is the answer supported
   by the retrieved context?) and relevance (does it actually answer the
   question?), scored 0-1 by an LLM judge.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.generation.generator import build_llm_client
from src.pipeline import RAGPipeline


@dataclass
class EvalResult:
    question: str
    expected_source: str | None
    retrieved_sources: list[str] = field(default_factory=list)
    hit: bool = False
    reciprocal_rank: float = 0.0
    answer: str = ""
    faithfulness: float | None = None
    relevance: float | None = None


def load_qa_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _score_retrieval(expected_source: str | None, retrieved_sources: list[str]) -> tuple[bool, float]:
    if not expected_source:
        return False, 0.0
    for rank, src in enumerate(retrieved_sources, start=1):
        if expected_source in src or src in expected_source:
            return True, 1.0 / rank
    return False, 0.0


JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer.

Question: {question}
Context provided to the model:
{context}

Model's answer: {answer}

Score the answer on two dimensions, each from 0.0 to 1.0:
- faithfulness: is every claim in the answer actually supported by the context? \
(1.0 = fully grounded, 0.0 = fabricated/unsupported)
- relevance: does the answer actually address the question asked? \
(1.0 = fully relevant, 0.0 = off-topic)

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <float>, "relevance": <float>}}
"""


def _judge_answer(question: str, context: str, answer: str) -> tuple[float | None, float | None]:
    try:
        client = build_llm_client()
        raw = client.generate(
            "You are a strict, impartial evaluator. Respond with valid JSON only.",
            JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer),
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None, None
        data = json.loads(match.group(0))
        return float(data.get("faithfulness")), float(data.get("relevance"))
    except Exception as e:  # noqa: BLE001 - evaluation should never crash the run
        print(f"[evaluator] judge scoring failed: {e}")
        return None, None


def run_evaluation(qa_path: str, judge: bool = True) -> dict:
    pipeline = RAGPipeline()
    qa_set = load_qa_set(qa_path)
    results: list[EvalResult] = []

    for item in qa_set:
        question = item["question"]
        expected_source = item.get("expected_source")

        retrieved = pipeline.retrieve_only(question)
        retrieved_sources = [rc.chunk.source for rc in retrieved]
        hit, rr = _score_retrieval(expected_source, retrieved_sources)

        result = EvalResult(
            question=question,
            expected_source=expected_source,
            retrieved_sources=retrieved_sources,
            hit=hit,
            reciprocal_rank=rr,
        )

        if judge and retrieved:
            gen_result = pipeline.generator.answer(question, retrieved)
            result.answer = gen_result["answer"]
            context = "\n\n".join(rc.chunk.text for rc in retrieved)
            result.faithfulness, result.relevance = _judge_answer(
                question, context, result.answer
            )

        results.append(result)

    n = len(results) or 1
    hit_rate = sum(r.hit for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    faith_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    rel_scores = [r.relevance for r in results if r.relevance is not None]

    summary = {
        "num_questions": len(results),
        "retrieval_hit_rate": round(hit_rate, 3),
        "mean_reciprocal_rank": round(mrr, 3),
        "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None,
        "avg_relevance": round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else None,
    }
    return {"summary": summary, "results": [r.__dict__ for r in results]}
