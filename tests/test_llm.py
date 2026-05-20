"""MockClient tests.

MockClient is the LLMClient used in every other test. If it is broken, the
test suite is meaningless.
"""
from __future__ import annotations

from multi_agent_supervisor.llm import MockClient


class TestMockClient:
    def test_returns_matched_response(self):
        client = MockClient(responses={"hello": "hi back"})
        result = client.complete("say hello please")
        assert result.text == "hi back"

    def test_returns_default_on_no_match(self):
        client = MockClient(default="fallback")
        result = client.complete("anything")
        assert result.text == "fallback"

    def test_counts_calls(self):
        client = MockClient()
        client.complete("first")
        client.complete("second")
        client.complete("third")
        assert client.call_count == 3
        assert client.last_prompts == ["first", "second", "third"]

    def test_first_match_wins(self):
        client = MockClient(responses={"abc": "first", "def": "second"})
        # both keys appear, "abc" should be matched first since dicts are insertion-ordered
        result = client.complete("abc and def")
        assert result.text == "first"

    def test_reports_token_counts(self):
        client = MockClient(default="some output")
        result = client.complete("a prompt that is somewhat longer")
        assert result.tokens_in > 0
        assert result.tokens_out > 0
