"""Tool layer tests: JSON parsing tolerance and the StaticSearcher."""
from __future__ import annotations

import pytest

from multi_agent_supervisor.state import RetrievedDoc
from multi_agent_supervisor.tools.wikipedia import StaticSearcher, parse_json_block


class TestParseJsonBlock:
    def test_parses_pure_json(self):
        assert parse_json_block('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert parse_json_block(text) == {"a": 1}

    def test_strips_unlabeled_fence(self):
        text = '```\n{"a": 1}\n```'
        assert parse_json_block(text) == {"a": 1}

    def test_extracts_embedded_object(self):
        text = 'Sure, here is the JSON:\n{"a": 1, "b": 2}\nLet me know if you need more.'
        assert parse_json_block(text) == {"a": 1, "b": 2}

    def test_extracts_embedded_array(self):
        text = "Result: [1, 2, 3] thanks"
        assert parse_json_block(text) == [1, 2, 3]

    def test_handles_nested_objects(self):
        text = '{"outer": {"inner": [1, 2]}, "flat": true}'
        result = parse_json_block(text)
        assert result["outer"]["inner"] == [1, 2]
        assert result["flat"] is True

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError):
            parse_json_block("no json here at all")


class TestStaticSearcher:
    def test_returns_corpus_match(self):
        searcher = StaticSearcher(
            corpus={
                "who is Ada Lovelace": [
                    RetrievedDoc(title="Ada Lovelace", snippet="English mathematician..."),
                ],
            }
        )
        docs = searcher.search("who is Ada Lovelace")
        assert len(docs) == 1
        assert docs[0].title == "Ada Lovelace"

    def test_returns_empty_on_miss(self):
        searcher = StaticSearcher(corpus={})
        assert searcher.search("anything") == []

    def test_respects_top_k(self):
        searcher = StaticSearcher(
            corpus={
                "query": [
                    RetrievedDoc(title="A", snippet="..."),
                    RetrievedDoc(title="B", snippet="..."),
                    RetrievedDoc(title="C", snippet="..."),
                ]
            }
        )
        assert len(searcher.search("query", top_k=2)) == 2
