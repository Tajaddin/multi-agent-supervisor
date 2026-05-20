"""Command-line entry point.

Usage:
    mas "Who directed the 2010 film Inception and what other films did they direct?"

Requires ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from multi_agent_supervisor.llm import AnthropicClient
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import WikipediaSearcher


def _format_state(state: dict) -> str:
    out = []
    out.append(f"\nFinal answer:\n{state.get('final_answer', '')}\n")
    cites = state.get("citations", []) or []
    if cites:
        out.append("Citations:")
        for c in cites:
            out.append(f"  - {c.title} ({c.url})")
    sub_qs = state.get("sub_questions", []) or []
    if sub_qs:
        out.append("\nSub-questions:")
        for sq in sub_qs:
            out.append(f"  - {sq.text}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent supervisor CLI.")
    parser.add_argument("query", help="The research question to answer.")
    parser.add_argument("--top-k", type=int, default=3, help="Docs per sub-question.")
    parser.add_argument("--max-sub", type=int, default=4, help="Max sub-questions.")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    llm = AnthropicClient(model=args.model)
    searcher = WikipediaSearcher()
    graph = build_supervisor(llm, searcher, top_k=args.top_k, max_sub_questions=args.max_sub)

    start = time.perf_counter()
    final_state = graph.invoke({"query": args.query})
    elapsed = time.perf_counter() - start

    if args.json:
        payload = {
            "query": args.query,
            "final_answer": final_state.get("final_answer", ""),
            "citations": [c.model_dump() for c in final_state.get("citations", []) or []],
            "sub_questions": [sq.model_dump() for sq in final_state.get("sub_questions", []) or []],
            "elapsed_seconds": elapsed,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_state(final_state))
        print(f"\nElapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
