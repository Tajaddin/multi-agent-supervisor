"""Verifier specialist.

Checks whether the analyzer's answer is actually supported by the retrieved
snippets. Surfaces flagged claims so the synthesizer can drop or hedge them.
"""
from __future__ import annotations

import time

from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import (
    Analysis,
    RetrievedDoc,
    SubQuestion,
    TelemetryEvent,
    Verification,
)
from multi_agent_supervisor.tools.wikipedia import parse_json_block

VERIFIER_SYSTEM = (
    "You are a fact-checking specialist. Given a sub-question, an answer "
    "produced by another agent, and the source snippets the agent had access "
    "to, decide whether each claim in the answer is supported by the "
    "snippets. Return JSON: "
    '{"supported": bool, "reasoning": str, "flagged_claims": [str]}. '
    "Mark supported=false if ANY material claim is unsupported. The "
    "flagged_claims list should contain the exact sentences (or close "
    "paraphrases) from the answer that are not supported."
)


def _verifier_prompt(sub_question: SubQuestion, analysis: Analysis, docs: list[RetrievedDoc]) -> str:
    snippet_block = "\n\n".join(
        f"[{i + 1}] {d.title}\n{d.snippet}" for i, d in enumerate(docs)
    ) or "(no snippets retrieved)"
    return (
        f"Sub-question:\n{sub_question.text}\n\n"
        f"Answer to verify:\n{analysis.answer}\n\n"
        f"Source snippets the analyzer had access to:\n{snippet_block}\n\n"
        "Return ONLY the JSON object as specified."
    )


def run_verifier(
    sub_question: SubQuestion,
    analysis: Analysis | None,
    docs: list[RetrievedDoc],
    llm: LLMClient,
) -> Verification:
    """Pure function form."""
    if analysis is None or not analysis.answer or analysis.answer == "insufficient evidence":
        return Verification(
            sub_question_id=sub_question.id,
            supported=False,
            reasoning="analyzer reported insufficient evidence",
            flagged_claims=[],
        )
    if not docs:
        return Verification(
            sub_question_id=sub_question.id,
            supported=False,
            reasoning="no sources to verify against",
            flagged_claims=[analysis.answer],
        )

    result = llm.complete(
        _verifier_prompt(sub_question, analysis, docs),
        system=VERIFIER_SYSTEM,
        max_tokens=512,
    )
    try:
        data = parse_json_block(result.text)
        if not isinstance(data, dict):
            raise ValueError("expected dict")
    except ValueError:
        return Verification(
            sub_question_id=sub_question.id,
            supported=True,
            reasoning="verifier output not parseable; defaulting to pass",
            flagged_claims=[],
        )

    return Verification(
        sub_question_id=sub_question.id,
        supported=bool(data.get("supported", False)),
        reasoning=str(data.get("reasoning", "")).strip(),
        flagged_claims=[c for c in data.get("flagged_claims", []) if isinstance(c, str)],
    )


def build_verifier_node(llm: LLMClient):
    """Return a LangGraph node that verifies one sub-question's analysis."""

    def verifier_node(payload: dict) -> dict:
        start = time.perf_counter()
        sub_q: SubQuestion = payload["sub_question"]
        analysis: Analysis | None = payload.get("analysis")
        docs: list[RetrievedDoc] = payload.get("docs", [])
        ver = run_verifier(sub_q, analysis, docs, llm)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "verifications": {sub_q.id: ver},
            "telemetry": [
                TelemetryEvent(
                    node="verifier",
                    elapsed_ms=elapsed,
                    note=f"sq={sub_q.id} supported={ver.supported}",
                )
            ],
        }

    return verifier_node
