"""Sequential multi-agent baseline.

Same specialist chain as the supervisor but executes the agents one-at-a-time
instead of in parallel via Send. This isolates the speedup contribution of
parallelism (the only difference from supervisor.py).

Why this matters: if sequential matches parallel on latency, the supervisor's
hero claim collapses. We need a side-by-side to prove the speedup is from
parallel fan-out, not from having more agents.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from multi_agent_supervisor.agents.analyzer import run_analyzer
from multi_agent_supervisor.agents.planner import run_planner
from multi_agent_supervisor.agents.retriever import run_retriever
from multi_agent_supervisor.agents.synthesizer import run_synthesizer
from multi_agent_supervisor.agents.verifier import run_verifier
from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import (
    Analysis,
    Citation,
    RetrievedDoc,
    SubQuestion,
    Verification,
)
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol


@dataclass
class SequentialResult:
    answer: str
    citations: list[Citation]
    latency_ms: float
    sub_questions: list[SubQuestion]
    retrievals: dict[str, list[RetrievedDoc]] = field(default_factory=dict)
    analyses: dict[str, Analysis] = field(default_factory=dict)
    verifications: dict[str, Verification] = field(default_factory=dict)


def run_sequential(
    query: str,
    llm: LLMClient,
    searcher: SearcherProtocol,
    *,
    top_k: int = 3,
    max_sub_questions: int = 4,
) -> SequentialResult:
    """Run all specialists serially. Identical functional output, no Send."""
    t0 = time.perf_counter()

    sub_qs = run_planner(query, llm, max_sub_questions=max_sub_questions)

    retrievals: dict[str, list[RetrievedDoc]] = {}
    for sq in sub_qs:
        retrievals[sq.id] = run_retriever(sq, searcher, top_k=top_k)

    analyses: dict[str, Analysis] = {}
    for sq in sub_qs:
        analyses[sq.id] = run_analyzer(sq, retrievals[sq.id], llm)

    verifications: dict[str, Verification] = {}
    for sq in sub_qs:
        verifications[sq.id] = run_verifier(sq, analyses[sq.id], retrievals[sq.id], llm)

    final_answer, citations = run_synthesizer(
        query, sub_qs, analyses, verifications, retrievals, llm
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    return SequentialResult(
        answer=final_answer,
        citations=citations,
        latency_ms=latency_ms,
        sub_questions=sub_qs,
        retrievals=retrievals,
        analyses=analyses,
        verifications=verifications,
    )
