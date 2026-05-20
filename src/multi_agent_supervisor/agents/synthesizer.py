"""Synthesizer: final step that merges per-sub-question analyses into one answer.

Runs once after every specialist has reported. Drops or hedges claims that the
verifier flagged as unsupported. Surfaces consolidated citations.
"""
from __future__ import annotations

import time

from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import (
    Analysis,
    Citation,
    RetrievedDoc,
    SubQuestion,
    SupervisorState,
    TelemetryEvent,
    Verification,
)
from multi_agent_supervisor.tools.wikipedia import parse_json_block

SYNTHESIZER_SYSTEM = (
    "You are the synthesizer. You receive a list of sub-questions, each with "
    "an analyzed answer and a verifier verdict. Produce a single concise "
    "final answer to the original user query. If a sub-answer was flagged as "
    "unsupported, drop or hedge those claims (use 'reportedly' or 'unclear'). "
    'Return JSON: {"final_answer": str, "used_citations": [str]}. '
    "Keep the answer under 6 sentences."
)


def _synthesizer_prompt(
    query: str,
    sub_questions: list[SubQuestion],
    analyses: dict[str, Analysis],
    verifications: dict[str, Verification],
) -> str:
    blocks = []
    for sq in sub_questions:
        a = analyses.get(sq.id)
        v = verifications.get(sq.id)
        a_text = a.answer if a is not None else "(no analysis)"
        v_text = (
            f"supported={v.supported}; flagged={v.flagged_claims}"
            if v is not None
            else "(no verification)"
        )
        cites = ", ".join(a.citations) if a is not None else ""
        blocks.append(
            f"Sub-question: {sq.text}\n"
            f"  Analyzer answer: {a_text}\n"
            f"  Citations: {cites}\n"
            f"  Verifier: {v_text}"
        )
    body = "\n\n".join(blocks) if blocks else "(no sub-questions)"
    return (
        f"Original user query:\n{query}\n\n"
        f"Sub-question results:\n{body}\n\n"
        "Return ONLY the JSON object as specified."
    )


def _consolidate_citations(
    sub_questions: list[SubQuestion],
    analyses: dict[str, Analysis],
    retrievals: dict[str, list[RetrievedDoc]],
    used_titles: list[str],
) -> list[Citation]:
    """Build the final citation list by looking up each used title in retrievals."""
    used_set = {t.lower() for t in used_titles}
    seen: set[str] = set()
    out: list[Citation] = []
    for sq in sub_questions:
        docs = retrievals.get(sq.id, [])
        for d in docs:
            key = d.title.lower()
            if key in seen:
                continue
            if used_titles and key not in used_set:
                continue
            seen.add(key)
            out.append(Citation(title=d.title, url=d.url, snippet=d.snippet))
    # Fall back: if the synthesizer named no citations, surface any cited by analyzers.
    if not out:
        used_from_analyses = {c.lower() for a in analyses.values() for c in a.citations}
        for sq in sub_questions:
            for d in retrievals.get(sq.id, []):
                key = d.title.lower()
                if key in seen:
                    continue
                if used_from_analyses and key not in used_from_analyses:
                    continue
                seen.add(key)
                out.append(Citation(title=d.title, url=d.url, snippet=d.snippet))
    return out


def run_synthesizer(
    query: str,
    sub_questions: list[SubQuestion],
    analyses: dict[str, Analysis],
    verifications: dict[str, Verification],
    retrievals: dict[str, list[RetrievedDoc]],
    llm: LLMClient,
) -> tuple[str, list[Citation]]:
    """Pure function form. Returns (final_answer, citations)."""
    if not sub_questions:
        return "No sub-questions were produced.", []

    result = llm.complete(
        _synthesizer_prompt(query, sub_questions, analyses, verifications),
        system=SYNTHESIZER_SYSTEM,
        max_tokens=800,
    )
    try:
        data = parse_json_block(result.text)
        if not isinstance(data, dict):
            raise ValueError("expected dict")
        final_answer = str(data.get("final_answer", "")).strip()
        used = [c for c in data.get("used_citations", []) if isinstance(c, str)]
    except ValueError:
        final_answer = result.text.strip()[:1000]
        used = []

    if not final_answer:
        final_answer = "No answer could be synthesized."

    citations = _consolidate_citations(sub_questions, analyses, retrievals, used)
    return final_answer, citations


def build_synthesizer_node(llm: LLMClient):
    """Return a LangGraph node that synthesizes the final answer."""

    def synthesizer_node(state: SupervisorState) -> dict:
        start = time.perf_counter()
        final_answer, citations = run_synthesizer(
            state.get("query", ""),
            state.get("sub_questions", []),
            state.get("analyses", {}),
            state.get("verifications", {}),
            state.get("retrievals", {}),
            llm,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "final_answer": final_answer,
            "citations": citations,
            "telemetry": [
                TelemetryEvent(
                    node="synthesizer",
                    elapsed_ms=elapsed,
                    note=f"citations={len(citations)}",
                )
            ],
        }

    return synthesizer_node
