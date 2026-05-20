"""Planner: decomposes the user query into atomic sub-questions.

Runs once at the top of the supervisor graph. Its only side effect is setting
state["sub_questions"]. The supervisor's fan-out router reads that list and
dispatches one specialist instance per entry.
"""
from __future__ import annotations

import time
import uuid

from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import SubQuestion, SupervisorState, TelemetryEvent
from multi_agent_supervisor.tools.wikipedia import parse_json_block

PLANNER_SYSTEM = (
    "You decompose a research question into 2 to 4 atomic sub-questions for "
    "downstream specialist agents. Each sub-question should be answerable by "
    "looking up a single Wikipedia article. Avoid yes/no questions. Avoid "
    "re-stating the original query verbatim. Return JSON with one key "
    '"sub_questions" mapping to a list of objects {"text": str, "rationale": str}.'
)


def _planner_prompt(query: str) -> str:
    return (
        f"User query:\n{query}\n\n"
        "Return ONLY a JSON object of the form:\n"
        '{"sub_questions": [{"text": "...", "rationale": "..."}, ...]}'
    )


def run_planner(query: str, llm: LLMClient, max_sub_questions: int = 4) -> list[SubQuestion]:
    """Pure function form of the planner, callable outside LangGraph for tests."""
    if not query.strip():
        return []
    result = llm.complete(_planner_prompt(query), system=PLANNER_SYSTEM, max_tokens=512)
    try:
        data = parse_json_block(result.text)
    except ValueError:
        # Fall back to a single sub-question matching the full query so the
        # downstream graph never stalls on a malformed model output.
        return [SubQuestion(id=f"sq-{uuid.uuid4().hex[:8]}", text=query.strip())]

    raw = data.get("sub_questions", []) if isinstance(data, dict) else []
    sub_qs: list[SubQuestion] = []
    for i, entry in enumerate(raw[:max_sub_questions]):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", "").strip()
        if not text:
            continue
        sub_qs.append(
            SubQuestion(
                id=f"sq-{i:02d}-{uuid.uuid4().hex[:6]}",
                text=text,
                rationale=entry.get("rationale", ""),
            )
        )
    if not sub_qs:
        return [SubQuestion(id=f"sq-{uuid.uuid4().hex[:8]}", text=query.strip())]
    return sub_qs


def build_planner_node(llm: LLMClient, max_sub_questions: int = 4):
    """Return a LangGraph node that runs the planner."""

    def planner_node(state: SupervisorState) -> dict:
        start = time.perf_counter()
        query = state.get("query", "")
        sub_qs = run_planner(query, llm, max_sub_questions=max_sub_questions)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "sub_questions": sub_qs,
            "telemetry": [
                TelemetryEvent(
                    node="planner",
                    elapsed_ms=elapsed,
                    note=f"emitted {len(sub_qs)} sub-questions",
                )
            ],
        }

    return planner_node
