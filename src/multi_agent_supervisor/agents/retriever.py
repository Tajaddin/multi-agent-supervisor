"""Retriever specialist.

Invoked once per sub-question via LangGraph's Send. Each instance runs the
configured Searcher against the sub-question text and writes its results into
state["retrievals"][sub_question.id]. The merge_dicts reducer on that field
keeps the writes from clobbering each other.
"""
from __future__ import annotations

import time

from multi_agent_supervisor.state import RetrievedDoc, SubQuestion, TelemetryEvent
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol


def run_retriever(sub_question: SubQuestion, searcher: SearcherProtocol, top_k: int = 3) -> list[RetrievedDoc]:
    """Pure function form, callable outside LangGraph."""
    return searcher.search(sub_question.text, top_k=top_k)


def build_retriever_node(searcher: SearcherProtocol, top_k: int = 3):
    """Return a LangGraph node that retrieves docs for one sub-question.

    The node consumes the Send payload (not the parent state). The payload
    must include a `sub_question` field.
    """

    def retriever_node(payload: dict) -> dict:
        start = time.perf_counter()
        sub_q: SubQuestion = payload["sub_question"]
        docs = run_retriever(sub_q, searcher, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "retrievals": {sub_q.id: docs},
            "telemetry": [
                TelemetryEvent(
                    node="retriever",
                    elapsed_ms=elapsed,
                    note=f"sq={sub_q.id} docs={len(docs)}",
                )
            ],
        }

    return retriever_node
