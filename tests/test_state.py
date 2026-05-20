"""Reducer tests.

The supervisor relies on merge_dicts and append_list to combine state from
parallel specialist invocations. If either reducer is broken, parallel writes
will overwrite each other and the benchmark will collapse to single-thread.
"""
from __future__ import annotations

from multi_agent_supervisor.state import (
    Analysis,
    RetrievedDoc,
    SubQuestion,
    TelemetryEvent,
    append_list,
    merge_dicts,
)


class TestMergeDicts:
    def test_merges_disjoint_keys(self):
        left = {"a": 1}
        right = {"b": 2}
        assert merge_dicts(left, right) == {"a": 1, "b": 2}

    def test_right_wins_on_conflict(self):
        assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}

    def test_handles_none_left(self):
        assert merge_dicts(None, {"a": 1}) == {"a": 1}

    def test_handles_none_right(self):
        assert merge_dicts({"a": 1}, None) == {"a": 1}

    def test_handles_both_none(self):
        assert merge_dicts(None, None) == {}

    def test_does_not_mutate_inputs(self):
        left = {"a": 1}
        right = {"b": 2}
        merge_dicts(left, right)
        assert left == {"a": 1}
        assert right == {"b": 2}

    def test_simulates_parallel_specialist_writes(self):
        # Two analyzers writing to different sub-question slots in parallel.
        a1 = {"sq-01": Analysis(sub_question_id="sq-01", answer="A")}
        a2 = {"sq-02": Analysis(sub_question_id="sq-02", answer="B")}
        merged = merge_dicts(a1, a2)
        assert set(merged.keys()) == {"sq-01", "sq-02"}
        assert merged["sq-01"].answer == "A"
        assert merged["sq-02"].answer == "B"


class TestAppendList:
    def test_appends(self):
        e1 = [TelemetryEvent(node="a", elapsed_ms=1.0)]
        e2 = [TelemetryEvent(node="b", elapsed_ms=2.0)]
        out = append_list(e1, e2)
        assert [e.node for e in out] == ["a", "b"]

    def test_handles_none(self):
        e1 = [TelemetryEvent(node="a", elapsed_ms=1.0)]
        assert len(append_list(e1, None)) == 1
        assert len(append_list(None, e1)) == 1
        assert append_list(None, None) == []


class TestSubQuestion:
    def test_minimal_fields(self):
        sq = SubQuestion(id="sq-01", text="Who is Ada Lovelace?")
        assert sq.id == "sq-01"
        assert sq.text == "Who is Ada Lovelace?"
        assert sq.rationale == ""


class TestRetrievedDoc:
    def test_minimal_fields(self):
        d = RetrievedDoc(title="Ada Lovelace", snippet="English mathematician...")
        assert d.title == "Ada Lovelace"
        assert d.url == ""
        assert d.score == 0.0
