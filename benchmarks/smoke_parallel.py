"""Smoke benchmark with no API key required.

Runs sequential and parallel modes against a SleepyClient that mimics a
realistic 0.7s per-LLM-call latency. Confirms the parallel supervisor delivers
the speedup advertised in the README.

Why this exists alongside hotpotqa_eval.py:
- hotpotqa_eval.py needs an ANTHROPIC_API_KEY and real Wikipedia access.
- This script runs in CI, in 6-7 seconds, with no secrets, and prints a
  reproducible speedup number any reader can verify in one command:

    python -m benchmarks.smoke_parallel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.baselines.sequential import run_sequential
from multi_agent_supervisor.llm import CompletionResult
from multi_agent_supervisor.state import RetrievedDoc
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import StaticSearcher

PLANNER = (
    '{"sub_questions": ['
    '{"text": "Who directed Inception?"},'
    '{"text": "Who composed the Inception score?"},'
    '{"text": "When was Inception released?"}'
    "]}"
)
ANALYZER = '{"answer": "A specialist answer.", "citations": ["Inception"], "confidence": 0.9}'
VERIFIER = '{"supported": true, "reasoning": "ok", "flagged_claims": []}'
SYNTH = '{"final_answer": "Synthesized from sub-answers.", "used_citations": ["Inception"]}'


@dataclass
class SleepyClient:
    delay: float = 0.7
    responses: dict[str, str] = field(default_factory=dict)
    default: str = ""
    call_count: int = 0

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        self.call_count += 1
        time.sleep(self.delay)
        for key, value in self.responses.items():
            if key in prompt:
                return CompletionResult(text=value, tokens_in=1, tokens_out=1)
        return CompletionResult(text=self.default, tokens_in=1, tokens_out=1)


def _build_corpus(n_sq: int) -> dict[str, list[RetrievedDoc]]:
    return {
        sq_text: [RetrievedDoc(title=f"Doc-{i}", url=f"u{i}", snippet=f"snippet-{i}")]
        for i, sq_text in enumerate(
            [
                "Who directed Inception?",
                "Who composed the Inception score?",
                "When was Inception released?",
            ][:n_sq]
        )
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel-vs-sequential smoke benchmark.")
    parser.add_argument("--delay", type=float, default=0.7, help="Per-LLM-call delay in seconds.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per mode.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "smoke_parallel.json",
    )
    args = parser.parse_args(argv)

    responses = {
        "User query": PLANNER,
        "Snippets:": ANALYZER,
        "Answer to verify": VERIFIER,
        "Sub-question results": SYNTH,
    }
    corpus = _build_corpus(n_sq=3)
    searcher = StaticSearcher(corpus=corpus)

    seq_times: list[float] = []
    par_times: list[float] = []

    for i in range(args.repeats):
        seq_client = SleepyClient(delay=args.delay, responses=responses)
        result = run_sequential("Tell me about Inception.", seq_client, searcher)
        seq_times.append(result.latency_ms / 1000)
        print(f"  [seq {i + 1}/{args.repeats}] {seq_times[-1]:.2f}s ({seq_client.call_count} LLM calls)")

    for i in range(args.repeats):
        par_client = SleepyClient(delay=args.delay, responses=responses)
        graph = build_supervisor(par_client, searcher)
        t0 = time.perf_counter()
        graph.invoke({"query": "Tell me about Inception."})
        par_times.append(time.perf_counter() - t0)
        print(f"  [par {i + 1}/{args.repeats}] {par_times[-1]:.2f}s ({par_client.call_count} LLM calls)")

    seq_mean = sum(seq_times) / len(seq_times)
    par_mean = sum(par_times) / len(par_times)
    speedup = seq_mean / par_mean

    summary = {
        "delay_per_llm_call_seconds": args.delay,
        "n_sub_questions": 3,
        "total_llm_calls_per_query": 8,  # 1 planner + 3 analyzers + 3 verifiers + 1 synth
        "repeats": args.repeats,
        "sequential_mean_seconds": round(seq_mean, 3),
        "parallel_mean_seconds": round(par_mean, 3),
        "speedup": round(speedup, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n=== Smoke benchmark summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSpeedup parallel vs sequential: {speedup:.2f}x")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
