"""LLM client abstraction.

Two implementations:
- AnthropicClient: real Anthropic API, used in production and benchmarks.
- MockClient: deterministic stub for unit tests with no network.

Both expose the same .complete(...) signature so agents do not branch on the
backend at call sites.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class CompletionResult:
    """Returned by every LLMClient.complete call."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    """Protocol all LLM clients follow.

    Implementations should be sync (we wrap parallelism at the graph level via
    LangGraph's Send fan-out, which already runs nodes concurrently).
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult: ...


class AnthropicClient:
    """Anthropic Messages API client.

    Defaults to claude-haiku-4-5-20251001 because the supervisor uses many
    short specialist calls and Haiku 4.5 hits the latency/cost sweet spot.
    The supervisor itself optionally uses Sonnet for the planner step.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        try:
            from anthropic import Anthropic  # imported lazily so tests do not require the package
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. pip install anthropic, or use MockClient for tests."
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Use MockClient in tests.")

        self._client = Anthropic(api_key=key)
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        text_parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        return CompletionResult(
            text="".join(text_parts),
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
        )


@dataclass
class MockClient:
    """Deterministic stub LLM.

    Looks up `prompt` against a `responses` table by substring match. The
    first matching key wins. Falls back to `default` (or empty string).

    Counts every call so tests can assert that fan-out actually invoked the
    LLM N times in parallel.
    """

    responses: dict[str, str] = field(default_factory=dict)
    default: str = ""
    call_count: int = 0
    last_prompts: list[str] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult:
        self.call_count += 1
        self.last_prompts.append(prompt)
        for key, value in self.responses.items():
            if key in prompt:
                return CompletionResult(text=value, tokens_in=len(prompt) // 4, tokens_out=len(value) // 4)
        return CompletionResult(text=self.default, tokens_in=len(prompt) // 4, tokens_out=len(self.default) // 4)
