"""Minimal end-to-end example.

Requires ANTHROPIC_API_KEY in the environment.

    python examples/basic_query.py
"""
from __future__ import annotations

import os
import sys
import time

from multi_agent_supervisor.llm import AnthropicClient
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import WikipediaSearcher

QUERY = "Who composed the score for the 2010 film Inception, and what year did they win their first Academy Award?"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Skipping.", file=sys.stderr)
        return 1

    llm = AnthropicClient()
    searcher = WikipediaSearcher()
    graph = build_supervisor(llm, searcher, top_k=3, max_sub_questions=3)

    print(f"Query: {QUERY}\n")
    t0 = time.perf_counter()
    state = graph.invoke({"query": QUERY})
    elapsed = time.perf_counter() - t0

    print("Final answer:")
    print(state.get("final_answer", ""))
    print()

    for sq in state.get("sub_questions", []) or []:
        analysis = state.get("analyses", {}).get(sq.id)
        verification = state.get("verifications", {}).get(sq.id)
        print(f"- {sq.text}")
        if analysis is not None:
            print(f"    answer: {analysis.answer}")
            print(f"    cites:  {analysis.citations}")
        if verification is not None:
            mark = "supported" if verification.supported else "FLAGGED"
            print(f"    verify: {mark} ({verification.reasoning[:80]})")
        print()

    print(f"Citations: {len(state.get('citations', []) or [])}")
    print(f"Elapsed:   {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
