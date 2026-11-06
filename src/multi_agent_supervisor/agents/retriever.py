"""Retriever specialist.

Invoked once per sub-question via LangGraph's Send. Each instance runs the
configured Searcher against the sub-question text and writes its results into
state["retrievals"][sub_question.id]. The merge_dicts reducer on that field
keeps the writes from clobbering each other.

A search that fails over HTTP (timeout, 429, 5xx) yields no docs for that
sub-question only. The analyzer then answers "insufficient evidence" for it and
the other branches finish normally.
"""
from __future__ import annotations

import time

import httpx

from multi_agent_supervisor.state import RetrievedDoc, SubQuestion, TelemetryEvent
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol


def run_retriever(sub_question: SubQuestion, searcher: SearcherProtocol, top_k: int = 3) -> list[RetrievedDoc]:
    """Pure function form, callable outside LangGraph."""
    return searcher.search(sub_question.text, top_k=top_k)


def retrieve_or_empty(
    sub_question: SubQuestion, searcher: SearcherProtocol, top_k: int = 3
) -> tuple[list[RetrievedDoc], str]:
    """run_retriever, but an HTTP failure returns no docs and a short error note."""
    try:
        return run_retriever(sub_question, searcher, top_k=top_k), ""
    except httpx.HTTPStatusError as exc:
        return [], f"HTTP {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return [], type(exc).__name__


def build_retriever_node(searcher: SearcherProtocol, top_k: int = 3):
    """Return a LangGraph node that retrieves docs for one sub-question.

    The node consumes the Send payload (not the parent state). The payload
    must include a `sub_question` field.
    """

    def retriever_node(payload: dict) -> dict:
        start = time.perf_counter()
        sub_q: SubQuestion = payload["sub_question"]
        docs, error = retrieve_or_empty(sub_q, searcher, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        note = f"sq={sub_q.id} docs={len(docs)}"
        if error:
            note += f" error={error}"
        return {
            "retrievals": {sub_q.id: docs},
            "telemetry": [
                TelemetryEvent(
                    node="retriever",
                    elapsed_ms=elapsed,
                    note=note,
                )
            ],
        }

    return retriever_node
