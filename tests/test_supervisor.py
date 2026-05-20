"""End-to-end supervisor graph tests.

These run the full LangGraph compile + invoke with a MockClient and a
StaticSearcher. They prove:
- The planner -> retrievers fan-out actually dispatches N parallel instances.
- The retrievals reducer merges per-sub-question keys without overwriting.
- The analyzer fan-out reads from the merged retrievals state.
- The verifier sees both retrievals and analyses.
- The synthesizer assembles a final answer from the merged state.
"""
from __future__ import annotations

from multi_agent_supervisor.llm import MockClient
from multi_agent_supervisor.state import RetrievedDoc
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import StaticSearcher

PLANNER_RESPONSE = (
    '{"sub_questions": ['
    '{"text": "Who directed Inception?", "rationale": "director"},'
    '{"text": "Who composed the Inception score?", "rationale": "composer"}'
    "]}"
)

ANALYZER_RESPONSE = (
    '{"answer": "Christopher Nolan was responsible.",'
    ' "citations": ["Inception"],'
    ' "confidence": 0.9}'
)

VERIFIER_RESPONSE = (
    '{"supported": true,'
    ' "reasoning": "Snippet supports the claim.",'
    ' "flagged_claims": []}'
)

SYNTH_RESPONSE = (
    '{"final_answer": "Christopher Nolan directed Inception; Hans Zimmer composed the score.",'
    ' "used_citations": ["Inception"]}'
)


def _build_test_graph():
    """Build a supervisor whose every prompt type maps to a deterministic reply."""
    llm = MockClient(
        responses={
            "User query": PLANNER_RESPONSE,
            "Snippets:": ANALYZER_RESPONSE,
            "Answer to verify": VERIFIER_RESPONSE,
            "Sub-question results": SYNTH_RESPONSE,
        }
    )
    searcher = StaticSearcher(
        corpus={
            "Who directed Inception?": [
                RetrievedDoc(
                    title="Inception",
                    url="https://en.wikipedia.org/wiki/Inception_(film)",
                    snippet="2010 film directed by Christopher Nolan.",
                ),
            ],
            "Who composed the Inception score?": [
                RetrievedDoc(
                    title="Hans Zimmer",
                    url="https://en.wikipedia.org/wiki/Hans_Zimmer",
                    snippet="German composer; scored Inception.",
                ),
            ],
        }
    )
    graph = build_supervisor(llm, searcher)
    return graph, llm, searcher


class TestSupervisorEndToEnd:
    def test_full_run_produces_final_answer(self):
        graph, llm, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        assert "final_answer" in final_state
        assert "Nolan" in final_state["final_answer"]

    def test_planner_emitted_both_sub_questions(self):
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        sub_qs = final_state.get("sub_questions", [])
        assert len(sub_qs) == 2
        texts = [sq.text for sq in sub_qs]
        assert any("director" in t.lower() or "directed" in t.lower() for t in texts)

    def test_retrievals_keyed_by_sub_question_id(self):
        """The merge_dicts reducer must keep one entry per sub_question.id."""
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        retrievals = final_state.get("retrievals", {})
        sub_qs = final_state["sub_questions"]
        assert len(retrievals) == len(sub_qs), "parallel retrievals clobbered each other"
        for sq in sub_qs:
            assert sq.id in retrievals

    def test_analyses_keyed_by_sub_question_id(self):
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        analyses = final_state.get("analyses", {})
        for sq in final_state["sub_questions"]:
            assert sq.id in analyses, f"analyzer missed {sq.id}"

    def test_verifications_keyed_by_sub_question_id(self):
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        verifications = final_state.get("verifications", {})
        for sq in final_state["sub_questions"]:
            assert sq.id in verifications

    def test_citations_populated(self):
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        cites = final_state.get("citations", [])
        assert cites, "expected at least one citation"

    def test_llm_invoked_for_every_specialist_layer(self):
        """1 planner + N analyzers + N verifiers + 1 synthesizer = 2N + 2 LLM calls."""
        graph, llm, _ = _build_test_graph()
        graph.invoke({"query": "Tell me about Inception."})
        # 1 planner + 2 analyzers + 2 verifiers + 1 synthesizer = 6
        assert llm.call_count == 6, f"expected 6 LLM calls, got {llm.call_count}"

    def test_telemetry_captures_all_node_invocations(self):
        graph, _, _ = _build_test_graph()
        final_state = graph.invoke({"query": "Tell me about Inception."})
        telemetry = final_state.get("telemetry", [])
        nodes = [e.node for e in telemetry]
        assert "planner" in nodes
        assert nodes.count("retriever") == 2
        assert nodes.count("analyzer") == 2
        assert nodes.count("verifier") == 2
        assert "synthesizer" in nodes


class TestSupervisorEdgeCases:
    def test_empty_corpus_yields_insufficient_evidence(self):
        """Retriever finds nothing; analyzer must still report something."""
        llm = MockClient(
            responses={
                "User query": PLANNER_RESPONSE,
                "Sub-question results": SYNTH_RESPONSE,
            },
            default="{}",
        )
        searcher = StaticSearcher(corpus={})
        graph = build_supervisor(llm, searcher)
        final_state = graph.invoke({"query": "Tell me about Inception."})
        analyses = final_state.get("analyses", {})
        for a in analyses.values():
            assert a.answer == "insufficient evidence"
            assert a.confidence == 0.0

    def test_single_sub_question_path(self):
        single_planner = '{"sub_questions": [{"text": "Who directed Inception?"}]}'
        llm = MockClient(
            responses={
                "User query": single_planner,
                "Snippets:": ANALYZER_RESPONSE,
                "Answer to verify": VERIFIER_RESPONSE,
                "Sub-question results": SYNTH_RESPONSE,
            }
        )
        searcher = StaticSearcher(
            corpus={
                "Who directed Inception?": [
                    RetrievedDoc(title="Inception", url="u", snippet="2010 Nolan film."),
                ]
            }
        )
        graph = build_supervisor(llm, searcher)
        final_state = graph.invoke({"query": "Tell me about Inception."})
        assert len(final_state["sub_questions"]) == 1
        assert "Nolan" in final_state["final_answer"]
