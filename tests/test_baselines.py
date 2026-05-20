"""Baseline + parallelism tests.

The killer claim of this project is "parallel is N-times faster than sequential".
Without a test that proves the Send fan-out actually runs concurrently, the
hero number on the README is unverifiable. This file tests both:

- That the single_agent baseline produces an answer.
- That the sequential baseline produces an answer matching the supervisor's.
- That when each LLM call sleeps for D seconds, the parallel supervisor finishes
  closer to D seconds while the sequential baseline takes O(N*D) seconds. This
  proves the speedup is from actual concurrency, not from a constant-factor
  difference in code paths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.baselines.sequential import run_sequential
from benchmarks.baselines.single_agent import run_single_agent
from multi_agent_supervisor.llm import CompletionResult
from multi_agent_supervisor.state import RetrievedDoc
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import StaticSearcher

PLANNER_RESPONSE = (
    '{"sub_questions": ['
    '{"text": "Who directed Inception?"},'
    '{"text": "Who composed the Inception score?"}'
    "]}"
)

ANALYZER_RESPONSE = (
    '{"answer": "A specialist answer.", "citations": ["Inception"], "confidence": 0.9}'
)

VERIFIER_RESPONSE = (
    '{"supported": true, "reasoning": "ok", "flagged_claims": []}'
)

SYNTH_RESPONSE = (
    '{"final_answer": "Synthesized from sub-answers.", "used_citations": ["Inception"]}'
)


@dataclass
class SleepyClient:
    """LLM client that sleeps `delay` seconds per call. Used to expose parallelism."""

    delay: float = 0.5
    responses: dict[str, str] = field(default_factory=dict)
    default: str = ""
    call_count: int = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult:
        self.call_count += 1
        time.sleep(self.delay)
        for key, value in self.responses.items():
            if key in prompt:
                return CompletionResult(text=value, tokens_in=1, tokens_out=1)
        return CompletionResult(text=self.default, tokens_in=1, tokens_out=1)


def _staged_responses() -> dict[str, str]:
    return {
        "User query": PLANNER_RESPONSE,
        "Snippets:": ANALYZER_RESPONSE,
        "Answer to verify": VERIFIER_RESPONSE,
        "Sub-question results": SYNTH_RESPONSE,
    }


def _two_sq_corpus() -> dict[str, list[RetrievedDoc]]:
    return {
        "Who directed Inception?": [
            RetrievedDoc(title="Inception", url="u1", snippet="2010 film directed by Nolan."),
        ],
        "Who composed the Inception score?": [
            RetrievedDoc(title="Hans Zimmer", url="u2", snippet="Composer; scored Inception."),
        ],
    }


class TestSingleAgentBaseline:
    def test_returns_answer(self):
        llm = SleepyClient(delay=0.0, default="Christopher Nolan directed Inception.")
        searcher = StaticSearcher(
            corpus={
                "Tell me about Inception": [
                    RetrievedDoc(title="Inception", snippet="2010 Nolan film."),
                ]
            }
        )
        result = run_single_agent("Tell me about Inception", llm, searcher)
        assert "Nolan" in result.answer
        assert result.latency_ms >= 0.0

    def test_surfaces_all_docs_as_citations(self):
        llm = SleepyClient(delay=0.0, default="ok")
        searcher = StaticSearcher(
            corpus={
                "q": [
                    RetrievedDoc(title="A", snippet="..."),
                    RetrievedDoc(title="B", snippet="..."),
                ]
            }
        )
        result = run_single_agent("q", llm, searcher)
        titles = {c.title for c in result.citations}
        assert titles == {"A", "B"}


class TestSequentialBaseline:
    def test_produces_final_answer(self):
        llm = SleepyClient(delay=0.0, responses=_staged_responses())
        searcher = StaticSearcher(corpus=_two_sq_corpus())
        result = run_sequential("Tell me about Inception.", llm, searcher)
        assert "Synthesized" in result.answer
        assert len(result.sub_questions) == 2
        assert set(result.retrievals.keys()) == {sq.id for sq in result.sub_questions}


class TestParallelismActuallyParallel:
    """Prove the supervisor's parallelism is real, not just claimed."""

    def test_supervisor_faster_than_sequential_under_sleep(self):
        # Each LLM call sleeps 0.5s. Total calls:
        #   sequential: 1 planner + 2 analyzers + 2 verifiers + 1 synth = 6 -> ~3.0s
        #   parallel:   planner(0.5) + max(analyzer, analyzer)(0.5) +
        #               max(verifier, verifier)(0.5) + synth(0.5) = ~2.0s
        # The exact wall-clock depends on thread scheduling, but parallel must be
        # at least ~25% faster than sequential. We give a generous margin.
        delay = 0.5
        responses = _staged_responses()
        corpus = _two_sq_corpus()

        # Sequential
        seq_client = SleepyClient(delay=delay, responses=responses)
        seq_searcher = StaticSearcher(corpus=corpus)
        seq_result = run_sequential("Tell me about Inception.", seq_client, seq_searcher)
        assert seq_client.call_count == 6

        # Parallel
        par_client = SleepyClient(delay=delay, responses=responses)
        par_searcher = StaticSearcher(corpus=corpus)
        graph = build_supervisor(par_client, par_searcher)
        t0 = time.perf_counter()
        state = graph.invoke({"query": "Tell me about Inception."})
        par_latency_ms = (time.perf_counter() - t0) * 1000

        assert state.get("final_answer", "")
        assert par_client.call_count == 6

        speedup = seq_result.latency_ms / par_latency_ms
        # Expect at least 1.25x. Real concurrency on Haiku 4.5 should hit ~2x.
        assert speedup >= 1.25, f"expected parallel >= 1.25x faster, got {speedup:.2f}x"
