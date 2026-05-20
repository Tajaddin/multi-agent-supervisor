"""Supervisor graph.

Topology:

    START
      |
    plan ----------------> fan-out via Send -> retrieve (x N parallel)
                                                      |
                                                      v   (barrier)
                                            dispatch_analysis
                                                      |
                                                      v
                                fan-out via Send -> analyze  (x N parallel)
                                                      |
                                                      v   (barrier)
                                            dispatch_verify
                                                      |
                                                      v
                                fan-out via Send -> verify   (x N parallel)
                                                      |
                                                      v   (barrier)
                                                synthesize
                                                      |
                                                      v
                                                     END

Each Send-spawned branch is a parallel node invocation. LangGraph batches all
sibling Sends into one superstep, then merges their state updates via the
reducers declared on SupervisorState. After every sibling finishes, the next
deterministic edge fires once with the merged state.
"""
from __future__ import annotations

from typing import Any

from langgraph.constants import Send
from langgraph.graph import StateGraph

from multi_agent_supervisor.agents.analyzer import build_analyzer_node
from multi_agent_supervisor.agents.planner import build_planner_node
from multi_agent_supervisor.agents.retriever import build_retriever_node
from multi_agent_supervisor.agents.synthesizer import build_synthesizer_node
from multi_agent_supervisor.agents.verifier import build_verifier_node
from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import SupervisorState
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol


def _dispatch_retrievals(state: SupervisorState) -> list[Send]:
    """Send one retriever instance per sub-question."""
    sub_qs = state.get("sub_questions", []) or []
    return [Send("retrieve", {"sub_question": sq}) for sq in sub_qs]


def _dispatch_analyzers(state: SupervisorState) -> list[Send]:
    """Send one analyzer instance per sub-question, passing its retrieved docs."""
    sub_qs = state.get("sub_questions", []) or []
    retrievals = state.get("retrievals", {}) or {}
    return [
        Send(
            "analyze",
            {
                "sub_question": sq,
                "docs": retrievals.get(sq.id, []),
            },
        )
        for sq in sub_qs
    ]


def _dispatch_verifiers(state: SupervisorState) -> list[Send]:
    """Send one verifier instance per sub-question, passing docs and analysis."""
    sub_qs = state.get("sub_questions", []) or []
    retrievals = state.get("retrievals", {}) or {}
    analyses = state.get("analyses", {}) or {}
    return [
        Send(
            "verify",
            {
                "sub_question": sq,
                "docs": retrievals.get(sq.id, []),
                "analysis": analyses.get(sq.id),
            },
        )
        for sq in sub_qs
    ]


def _passthrough(state: SupervisorState) -> dict:
    """No-op node that exists so Send fan-outs can converge before the next router."""
    return {}


def build_supervisor(
    llm: LLMClient,
    searcher: SearcherProtocol,
    *,
    top_k: int = 3,
    max_sub_questions: int = 4,
) -> Any:
    """Construct and compile the supervisor LangGraph.

    Parameters
    ----------
    llm
        LLMClient used by planner, analyzer, verifier, and synthesizer.
    searcher
        SearcherProtocol implementation used by the retriever specialist.
    top_k
        How many documents the retriever returns per sub-question.
    max_sub_questions
        Cap on how many sub-questions the planner emits.

    Returns
    -------
    Compiled LangGraph runnable. Call .invoke({"query": "..."}) on it.
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("plan", build_planner_node(llm, max_sub_questions=max_sub_questions))
    graph.add_node("retrieve", build_retriever_node(searcher, top_k=top_k))
    graph.add_node("dispatch_analysis", _passthrough)
    graph.add_node("analyze", build_analyzer_node(llm))
    graph.add_node("dispatch_verify", _passthrough)
    graph.add_node("verify", build_verifier_node(llm))
    graph.add_node("synthesize", build_synthesizer_node(llm))

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _dispatch_retrievals, ["retrieve"])
    graph.add_edge("retrieve", "dispatch_analysis")
    graph.add_conditional_edges("dispatch_analysis", _dispatch_analyzers, ["analyze"])
    graph.add_edge("analyze", "dispatch_verify")
    graph.add_conditional_edges("dispatch_verify", _dispatch_verifiers, ["verify"])
    graph.add_edge("verify", "synthesize")
    graph.set_finish_point("synthesize")

    return graph.compile()
