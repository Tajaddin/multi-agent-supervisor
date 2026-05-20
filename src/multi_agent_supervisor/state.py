"""Shared state for the multi-agent supervisor graph.

The state is a TypedDict consumed by LangGraph. Per-sub-question fields use a
dict-merge reducer so parallel specialists writing into the same field do not
clobber each other.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge two dicts. Right wins on key conflict.

    Used as a LangGraph reducer so parallel agents writing per-sub-question
    results do not overwrite each other's keys.
    """
    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    out = dict(left)
    out.update(right)
    return out


def append_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Concatenate two lists. Used for telemetry events."""
    if left is None:
        left = []
    if right is None:
        right = []
    return [*left, *right]


class SubQuestion(BaseModel):
    """A decomposed sub-question produced by the planner."""

    id: str
    text: str
    rationale: str = ""


class RetrievedDoc(BaseModel):
    """One document/snippet retrieved by the retriever specialist."""

    title: str
    url: str = ""
    snippet: str
    score: float = 0.0


class Analysis(BaseModel):
    """Per-sub-question synthesis from the analyzer specialist."""

    sub_question_id: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Verification(BaseModel):
    """Per-sub-question fact-check from the verifier specialist."""

    sub_question_id: str
    supported: bool
    reasoning: str
    flagged_claims: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A final citation surfaced in the synthesized answer."""

    title: str
    url: str = ""
    snippet: str = ""


class TelemetryEvent(BaseModel):
    """One observable event during a supervisor run."""

    node: str
    elapsed_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    note: str = ""


class SupervisorState(TypedDict, total=False):
    """LangGraph state for the multi-agent supervisor.

    All per-sub-question fields use merge_dicts as the reducer so parallel
    specialist invocations (one Send() per sub-question) can update the same
    field concurrently without losing each other's writes.
    """

    query: str
    sub_questions: list[SubQuestion]

    retrievals: Annotated[dict[str, list[RetrievedDoc]], merge_dicts]
    analyses: Annotated[dict[str, Analysis], merge_dicts]
    verifications: Annotated[dict[str, Verification], merge_dicts]

    final_answer: str
    citations: list[Citation]

    telemetry: Annotated[list[TelemetryEvent], append_list]


class HandoffCommand(BaseModel):
    """Payload sent from the supervisor to a specialist via LangGraph Send.

    The specialist consumes its sub_question and writes back into the merged
    dict fields on SupervisorState. The supervisor never reads back from a
    single specialist directly; it inspects merged state after a barrier.
    """

    sub_question: SubQuestion
    upstream_retrievals: list[RetrievedDoc] = Field(default_factory=list)
    upstream_analysis: Analysis | None = None
