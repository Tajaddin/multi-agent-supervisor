"""Analyzer specialist.

Reads retrieved snippets for one sub-question and writes a synthesized
Analysis with citations. Runs once per sub-question via Send.
"""
from __future__ import annotations

import time

from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import Analysis, RetrievedDoc, SubQuestion, TelemetryEvent
from multi_agent_supervisor.tools.wikipedia import parse_json_block

ANALYZER_SYSTEM = (
    "You are a specialist analyzer. Given a sub-question and a small set of "
    "Wikipedia snippets, write a single-paragraph answer (1-3 sentences). "
    "Cite every claim by the source title in square brackets like [Article Title]. "
    "If the snippets do not support an answer, say 'insufficient evidence' and "
    "set confidence below 0.3. Return JSON: "
    '{"answer": str, "citations": [str], "confidence": float}'
)


def _analyzer_prompt(sub_question: SubQuestion, docs: list[RetrievedDoc]) -> str:
    snippet_block = "\n\n".join(
        f"[{i + 1}] {d.title}\n{d.snippet}" for i, d in enumerate(docs)
    ) or "(no snippets retrieved)"
    return (
        f"Sub-question:\n{sub_question.text}\n\n"
        f"Snippets:\n{snippet_block}\n\n"
        "Return ONLY a JSON object as specified."
    )


def run_analyzer(sub_question: SubQuestion, docs: list[RetrievedDoc], llm: LLMClient) -> Analysis:
    """Pure function form."""
    if not docs:
        return Analysis(
            sub_question_id=sub_question.id,
            answer="insufficient evidence",
            citations=[],
            confidence=0.0,
        )
    result = llm.complete(
        _analyzer_prompt(sub_question, docs),
        system=ANALYZER_SYSTEM,
        max_tokens=512,
    )
    try:
        data = parse_json_block(result.text)
        if not isinstance(data, dict):
            raise ValueError("expected dict")
    except ValueError:
        return Analysis(
            sub_question_id=sub_question.id,
            answer=result.text.strip()[:500],
            citations=[d.title for d in docs[:1]],
            confidence=0.3,
        )

    return Analysis(
        sub_question_id=sub_question.id,
        answer=str(data.get("answer", "")).strip() or "insufficient evidence",
        citations=[c for c in data.get("citations", []) if isinstance(c, str)],
        confidence=float(data.get("confidence", 0.5)),
    )


def build_analyzer_node(llm: LLMClient):
    """Return a LangGraph node that analyzes one sub-question's retrievals."""

    def analyzer_node(payload: dict) -> dict:
        start = time.perf_counter()
        sub_q: SubQuestion = payload["sub_question"]
        docs: list[RetrievedDoc] = payload.get("docs", [])
        analysis = run_analyzer(sub_q, docs, llm)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "analyses": {sub_q.id: analysis},
            "telemetry": [
                TelemetryEvent(
                    node="analyzer",
                    elapsed_ms=elapsed,
                    note=f"sq={sub_q.id} confidence={analysis.confidence:.2f}",
                )
            ],
        }

    return analyzer_node
