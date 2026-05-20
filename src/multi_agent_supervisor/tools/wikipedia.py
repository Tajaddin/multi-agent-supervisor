"""Wikipedia retrieval tool.

Uses the public MediaWiki REST API. No API key required.

The retriever specialist calls this with one sub-question per invocation. We
return up to top_k extracts, each truncated to ~400 characters so the
analyzer's prompt stays bounded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import httpx

from multi_agent_supervisor.state import RetrievedDoc


class SearcherProtocol(Protocol):
    """Pluggable retriever for tests."""

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedDoc]: ...


@dataclass
class WikipediaSearcher:
    """Hits Wikipedia search + extract APIs over HTTPS.

    Two calls per query: one to opensearch (titles), one to query/extracts
    (intro text). We keep the snippet under ~400 chars to bound analyzer
    prompt size.
    """

    base_url: str = "https://en.wikipedia.org/w/api.php"
    timeout: float = 8.0
    snippet_chars: int = 400
    user_agent: str = "multi-agent-supervisor/0.1 (https://github.com/Tajaddin/multi-agent-supervisor)"

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedDoc]:
        if not query.strip():
            return []
        headers = {"User-Agent": self.user_agent}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            titles = self._opensearch(client, query, top_k=top_k)
            if not titles:
                return []
            extracts = self._extracts(client, titles)
        docs = []
        for title in titles:
            snippet = extracts.get(title, "")[: self.snippet_chars]
            if not snippet:
                continue
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            docs.append(RetrievedDoc(title=title, url=url, snippet=snippet, score=1.0))
        return docs

    def _opensearch(self, client: httpx.Client, query: str, *, top_k: int) -> list[str]:
        params = {
            "action": "opensearch",
            "search": query,
            "limit": top_k,
            "namespace": 0,
            "format": "json",
        }
        resp = client.get(self.base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return []
        return list(data[1])

    def _extracts(self, client: httpx.Client, titles: list[str]) -> dict[str, str]:
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": "|".join(titles),
            "format": "json",
            "redirects": 1,
        }
        resp = client.get(self.base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        # Wikipedia returns a normalized title list that may differ from input
        # (redirects, case). Map back via the "normalized" / "redirects" fields.
        title_map = {}
        for entry in data.get("query", {}).get("normalized", []):
            title_map[entry["to"]] = entry["from"]
        for entry in data.get("query", {}).get("redirects", []):
            title_map[entry["to"]] = entry["from"]

        out = {}
        for page in pages.values():
            api_title = page.get("title", "")
            orig = title_map.get(api_title, api_title)
            text = page.get("extract", "") or ""
            if text:
                out[orig] = text
                # also key by canonical so callers using normalized titles still hit
                out[api_title] = text
        return out


@dataclass
class StaticSearcher:
    """Deterministic searcher for tests.

    Returns whatever is in `corpus[query]`. Falls back to empty list.
    """

    corpus: dict[str, list[RetrievedDoc]]

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedDoc]:
        return list(self.corpus.get(query, []))[:top_k]


def parse_json_block(text: str) -> dict | list:
    """Tolerantly extract the first JSON object/array from `text`.

    Models sometimes wrap JSON in markdown fences or add narration before/after.
    We grab the first { ... } or [ ... ] balanced span and json.loads it.
    Raises ValueError if nothing parsable is found.
    """
    text = text.strip()
    # Strip a leading ```json or ``` fence if present.
    if text.startswith("```"):
        # Drop the first fence line and the closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    # Try the whole string first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find the first balanced { ... } or [ ... ] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parsable JSON found in model output: {text[:200]!r}")
