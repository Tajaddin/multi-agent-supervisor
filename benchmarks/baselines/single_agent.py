"""Single-agent baseline.

One LLM call. We retrieve top-K Wikipedia docs against the raw query (no
decomposition) and concatenate them into one prompt. This is the cheapest,
fastest baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from multi_agent_supervisor.llm import LLMClient
from multi_agent_supervisor.state import Citation, RetrievedDoc
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol

SINGLE_AGENT_SYSTEM = (
    "Answer the user's question in 1-3 sentences. Cite each Wikipedia title "
    "you used in brackets like [Article Title]. If the snippets are "
    "insufficient, say so explicitly. Do not invent facts."
)


@dataclass
class SingleAgentResult:
    answer: str
    citations: list[Citation]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    retrievals: list[RetrievedDoc]


def run_single_agent(
    query: str,
    llm: LLMClient,
    searcher: SearcherProtocol,
    *,
    top_k: int = 6,
) -> SingleAgentResult:
    """Run the single-agent baseline on one query."""
    t0 = time.perf_counter()

    docs = searcher.search(query, top_k=top_k)
    snippet_block = "\n\n".join(
        f"[{i + 1}] {d.title}\n{d.snippet}" for i, d in enumerate(docs)
    ) or "(no snippets retrieved)"

    prompt = f"Question:\n{query}\n\nSnippets:\n{snippet_block}"

    result = llm.complete(prompt, system=SINGLE_AGENT_SYSTEM, max_tokens=400)

    latency_ms = (time.perf_counter() - t0) * 1000

    # The single-agent baseline does not parse citations; surface every doc as a
    # potential citation since the model had access to all of them.
    cites = [Citation(title=d.title, url=d.url, snippet=d.snippet) for d in docs]

    return SingleAgentResult(
        answer=result.text.strip(),
        citations=cites,
        latency_ms=latency_ms,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        retrievals=docs,
    )
