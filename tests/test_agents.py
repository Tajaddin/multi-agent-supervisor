"""Specialist agent tests.

Each agent is tested in two layers:
- The pure run_* function with a MockClient (no network, deterministic).
- The build_*_node wrapper to confirm it returns the right state delta shape.
"""
from __future__ import annotations

from multi_agent_supervisor.agents.analyzer import build_analyzer_node, run_analyzer
from multi_agent_supervisor.agents.planner import build_planner_node, run_planner
from multi_agent_supervisor.agents.retriever import build_retriever_node, run_retriever
from multi_agent_supervisor.agents.synthesizer import build_synthesizer_node, run_synthesizer
from multi_agent_supervisor.agents.verifier import run_verifier
from multi_agent_supervisor.llm import MockClient
from multi_agent_supervisor.state import Analysis, RetrievedDoc, SubQuestion
from multi_agent_supervisor.tools.wikipedia import StaticSearcher

PLANNER_RESPONSE = (
    '{"sub_questions": ['
    '{"text": "Who directed Inception?", "rationale": "identify director"},'
    '{"text": "What other films has that director directed?", "rationale": "filmography"}'
    "]}"
)


class TestPlanner:
    def test_emits_listed_sub_questions(self):
        llm = MockClient(responses={"User query": PLANNER_RESPONSE})
        sub_qs = run_planner("Tell me about Inception's director.", llm)
        assert len(sub_qs) == 2
        assert "Inception" in sub_qs[0].text
        assert sub_qs[0].rationale == "identify director"

    def test_caps_at_max_sub_questions(self):
        big = '{"sub_questions": [' + ",".join(
            f'{{"text": "q{i}"}}' for i in range(10)
        ) + "]}"
        llm = MockClient(responses={"User query": big})
        sub_qs = run_planner("anything", llm, max_sub_questions=3)
        assert len(sub_qs) == 3

    def test_falls_back_when_model_returns_garbage(self):
        llm = MockClient(default="this is not json")
        sub_qs = run_planner("How does photosynthesis work?", llm)
        assert len(sub_qs) == 1
        assert sub_qs[0].text == "How does photosynthesis work?"

    def test_empty_query_yields_no_sub_questions(self):
        llm = MockClient(default=PLANNER_RESPONSE)
        assert run_planner("   ", llm) == []

    def test_node_returns_state_delta(self):
        llm = MockClient(default=PLANNER_RESPONSE)
        node = build_planner_node(llm)
        delta = node({"query": "Tell me about Inception's director."})
        assert "sub_questions" in delta
        assert "telemetry" in delta
        assert delta["telemetry"][0].node == "planner"


class TestRetriever:
    def test_returns_corpus_docs(self):
        searcher = StaticSearcher(
            corpus={
                "Ada Lovelace": [
                    RetrievedDoc(title="Ada Lovelace", snippet="English mathematician..."),
                ],
            }
        )
        sq = SubQuestion(id="sq-01", text="Ada Lovelace")
        docs = run_retriever(sq, searcher)
        assert len(docs) == 1
        assert docs[0].title == "Ada Lovelace"

    def test_node_writes_under_sub_question_id(self):
        searcher = StaticSearcher(
            corpus={"q": [RetrievedDoc(title="T", snippet="S")]},
        )
        node = build_retriever_node(searcher)
        sq = SubQuestion(id="sq-42", text="q")
        delta = node({"sub_question": sq})
        assert "retrievals" in delta
        assert "sq-42" in delta["retrievals"]
        assert len(delta["retrievals"]["sq-42"]) == 1


ANALYZER_RESPONSE = (
    '{"answer": "Christopher Nolan directed Inception.",'
    ' "citations": ["Inception"],'
    ' "confidence": 0.9}'
)


class TestAnalyzer:
    def test_synthesizes_from_docs(self):
        llm = MockClient(responses={"Sub-question": ANALYZER_RESPONSE})
        sq = SubQuestion(id="sq-01", text="Who directed Inception?")
        docs = [RetrievedDoc(title="Inception", snippet="2010 film directed by Christopher Nolan.")]
        a = run_analyzer(sq, docs, llm)
        assert a.sub_question_id == "sq-01"
        assert "Nolan" in a.answer
        assert a.citations == ["Inception"]
        assert a.confidence == 0.9

    def test_short_circuits_when_no_docs(self):
        llm = MockClient(default=ANALYZER_RESPONSE)
        sq = SubQuestion(id="sq-01", text="anything")
        a = run_analyzer(sq, [], llm)
        assert a.answer == "insufficient evidence"
        assert a.confidence == 0.0
        # The LLM should NOT have been invoked when docs is empty.
        assert llm.call_count == 0

    def test_falls_back_when_model_returns_garbage(self):
        llm = MockClient(default="plain text answer with no JSON")
        sq = SubQuestion(id="sq-01", text="anything")
        docs = [RetrievedDoc(title="Source", snippet="some text")]
        a = run_analyzer(sq, docs, llm)
        assert a.answer.startswith("plain text answer")

    def test_node_returns_state_delta(self):
        llm = MockClient(default=ANALYZER_RESPONSE)
        node = build_analyzer_node(llm)
        sq = SubQuestion(id="sq-01", text="Who directed Inception?")
        docs = [RetrievedDoc(title="Inception", snippet="2010 film by Nolan.")]
        delta = node({"sub_question": sq, "docs": docs})
        assert "analyses" in delta
        assert "sq-01" in delta["analyses"]


VERIFIER_RESPONSE_SUPPORTED = (
    '{"supported": true,'
    ' "reasoning": "Snippet directly states Nolan directed.",'
    ' "flagged_claims": []}'
)

VERIFIER_RESPONSE_UNSUPPORTED = (
    '{"supported": false,'
    ' "reasoning": "Claim about budget not in snippets.",'
    ' "flagged_claims": ["budget was $160M"]}'
)


class TestVerifier:
    def test_marks_supported_when_model_agrees(self):
        llm = MockClient(default=VERIFIER_RESPONSE_SUPPORTED)
        sq = SubQuestion(id="sq-01", text="Who directed Inception?")
        analysis = Analysis(
            sub_question_id="sq-01",
            answer="Nolan directed Inception.",
            citations=["Inception"],
        )
        docs = [RetrievedDoc(title="Inception", snippet="2010 film directed by Christopher Nolan.")]
        v = run_verifier(sq, analysis, docs, llm)
        assert v.supported is True
        assert v.flagged_claims == []

    def test_flags_claims_when_model_disagrees(self):
        llm = MockClient(default=VERIFIER_RESPONSE_UNSUPPORTED)
        sq = SubQuestion(id="sq-01", text="Tell me about Inception.")
        analysis = Analysis(
            sub_question_id="sq-01",
            answer="Inception is a 2010 film, budget was $160M.",
            citations=["Inception"],
        )
        docs = [RetrievedDoc(title="Inception", snippet="2010 film.")]
        v = run_verifier(sq, analysis, docs, llm)
        assert v.supported is False
        assert "budget was $160M" in v.flagged_claims

    def test_short_circuits_when_no_analysis(self):
        llm = MockClient(default=VERIFIER_RESPONSE_SUPPORTED)
        sq = SubQuestion(id="sq-01", text="anything")
        v = run_verifier(sq, None, [], llm)
        assert v.supported is False
        assert llm.call_count == 0

    def test_short_circuits_on_insufficient_evidence(self):
        llm = MockClient(default=VERIFIER_RESPONSE_SUPPORTED)
        sq = SubQuestion(id="sq-01", text="anything")
        a = Analysis(sub_question_id="sq-01", answer="insufficient evidence")
        v = run_verifier(sq, a, [], llm)
        assert v.supported is False
        assert llm.call_count == 0


SYNTH_RESPONSE = (
    '{"final_answer": "Inception (2010) was directed by Christopher Nolan, who also directed Interstellar and Oppenheimer.",'
    ' "used_citations": ["Inception", "Christopher Nolan"]}'
)


class TestSynthesizer:
    def test_returns_final_answer_and_citations(self):
        llm = MockClient(default=SYNTH_RESPONSE)
        sub_qs = [
            SubQuestion(id="sq-01", text="Who directed Inception?"),
            SubQuestion(id="sq-02", text="What other films did they direct?"),
        ]
        analyses = {
            "sq-01": Analysis(sub_question_id="sq-01", answer="Nolan.", citations=["Inception"]),
            "sq-02": Analysis(
                sub_question_id="sq-02",
                answer="Interstellar and Oppenheimer.",
                citations=["Christopher Nolan"],
            ),
        }
        retrievals = {
            "sq-01": [RetrievedDoc(title="Inception", snippet="...", url="u1")],
            "sq-02": [RetrievedDoc(title="Christopher Nolan", snippet="...", url="u2")],
        }
        final, cites = run_synthesizer(
            "Tell me about Inception's director.",
            sub_qs,
            analyses,
            {},
            retrievals,
            llm,
        )
        assert "Nolan" in final
        titles = {c.title for c in cites}
        assert "Inception" in titles
        assert "Christopher Nolan" in titles

    def test_handles_empty_sub_questions(self):
        llm = MockClient(default=SYNTH_RESPONSE)
        final, cites = run_synthesizer("query", [], {}, {}, {}, llm)
        assert "no sub-questions" in final.lower()
        assert cites == []
        assert llm.call_count == 0

    def test_node_returns_state_delta(self):
        llm = MockClient(default=SYNTH_RESPONSE)
        node = build_synthesizer_node(llm)
        state = {
            "query": "q",
            "sub_questions": [SubQuestion(id="sq-01", text="x")],
            "analyses": {
                "sq-01": Analysis(sub_question_id="sq-01", answer="ans", citations=["A"]),
            },
            "verifications": {},
            "retrievals": {"sq-01": [RetrievedDoc(title="A", snippet="...")]},
        }
        delta = node(state)
        assert "final_answer" in delta
        assert "citations" in delta
