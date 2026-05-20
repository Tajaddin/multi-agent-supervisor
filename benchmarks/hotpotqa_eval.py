"""HotpotQA benchmark driver.

Runs three modes on a small dev-set subset:
  1. single_agent baseline (no decomposition)
  2. sequential multi-agent (specialists run serially)
  3. parallel supervisor (Send fan-out)

For each query, captures: F1 vs gold, EM, latency, token counts, citations.
Writes a JSON dump under benchmarks/results/ and a per-question markdown table.

Usage:
    python -m benchmarks.hotpotqa_eval --n 30 --seed 7
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.baselines.sequential import run_sequential
from benchmarks.baselines.single_agent import run_single_agent
from multi_agent_supervisor.llm import AnthropicClient, LLMClient
from multi_agent_supervisor.supervisor import build_supervisor
from multi_agent_supervisor.tools.wikipedia import SearcherProtocol, WikipediaSearcher

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class QAItem:
    question: str
    answer: str
    qid: str


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization. Lowercase, strip punctuation/articles/whitespace."""
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def contains_gold(prediction: str, gold: str) -> float:
    """Lenient match: does the prediction contain the gold answer as a substring?"""
    return float(normalize_answer(gold) in normalize_answer(prediction))


def load_hotpotqa_subset(n: int, seed: int = 7) -> list[QAItem]:
    """Load a deterministic subset of HotpotQA distractor dev.

    We hit the public datasets library so callers do not need to download by hand.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets package required. pip install 'multi-agent-supervisor[eval]'."
        ) from exc

    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    items = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        items.append(QAItem(question=ex["question"], answer=ex["answer"], qid=ex["id"]))
    return items


def run_one(
    mode: str,
    item: QAItem,
    llm: LLMClient,
    searcher: SearcherProtocol,
) -> dict[str, Any]:
    """Run one query under one mode. Returns a stat dict."""
    if mode == "single":
        result = run_single_agent(item.question, llm, searcher, top_k=6)
        return {
            "mode": mode,
            "qid": item.qid,
            "question": item.question,
            "gold": item.answer,
            "prediction": result.answer,
            "latency_ms": result.latency_ms,
            "citations": [c.title for c in result.citations],
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
        }
    if mode == "sequential":
        result = run_sequential(item.question, llm, searcher)
        return {
            "mode": mode,
            "qid": item.qid,
            "question": item.question,
            "gold": item.answer,
            "prediction": result.answer,
            "latency_ms": result.latency_ms,
            "citations": [c.title for c in result.citations],
            "n_sub_questions": len(result.sub_questions),
        }
    if mode == "parallel":
        graph = build_supervisor(llm, searcher)
        t0 = time.perf_counter()
        state = graph.invoke({"query": item.question})
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "mode": mode,
            "qid": item.qid,
            "question": item.question,
            "gold": item.answer,
            "prediction": state.get("final_answer", ""),
            "latency_ms": elapsed,
            "citations": [c.title for c in state.get("citations", []) or []],
            "n_sub_questions": len(state.get("sub_questions", []) or []),
        }
    raise ValueError(f"unknown mode: {mode}")


def score_record(rec: dict[str, Any]) -> dict[str, Any]:
    rec["f1"] = f1_score(rec["prediction"], rec["gold"])
    rec["em"] = exact_match(rec["prediction"], rec["gold"])
    rec["contains_gold"] = contains_gold(rec["prediction"], rec["gold"])
    return rec


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "n": len(records),
        "f1_mean": sum(r["f1"] for r in records) / len(records),
        "em_mean": sum(r["em"] for r in records) / len(records),
        "contains_gold_mean": sum(r["contains_gold"] for r in records) / len(records),
        "latency_ms_mean": sum(r["latency_ms"] for r in records) / len(records),
        "latency_ms_p50": sorted(r["latency_ms"] for r in records)[len(records) // 2],
        "latency_ms_max": max(r["latency_ms"] for r in records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HotpotQA bench for the multi-agent supervisor.")
    parser.add_argument("--n", type=int, default=30, help="Number of HotpotQA dev questions.")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single", "sequential", "parallel"],
        choices=["single", "sequential", "parallel"],
        help="Which baselines and the supervisor to run.",
    )
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "hotpotqa_results.json",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading HotpotQA distractor dev (n={args.n}, seed={args.seed})...", file=sys.stderr)
    items = load_hotpotqa_subset(args.n, seed=args.seed)
    print(f"Loaded {len(items)} questions.", file=sys.stderr)

    llm = AnthropicClient(model=args.model)
    searcher = WikipediaSearcher()

    all_records: list[dict[str, Any]] = []
    for mode in args.modes:
        print(f"\n=== Mode: {mode} ===", file=sys.stderr)
        for i, item in enumerate(items):
            print(f"  [{mode} {i + 1}/{len(items)}] {item.question[:80]}", file=sys.stderr)
            try:
                rec = run_one(mode, item, llm, searcher)
                rec = score_record(rec)
            except Exception as exc:  # noqa: BLE001 - report failure, keep going
                rec = {
                    "mode": mode,
                    "qid": item.qid,
                    "question": item.question,
                    "gold": item.answer,
                    "prediction": "",
                    "latency_ms": 0.0,
                    "f1": 0.0,
                    "em": 0.0,
                    "contains_gold": 0.0,
                    "error": repr(exc),
                }
            all_records.append(rec)

    summary = {mode: aggregate([r for r in all_records if r["mode"] == mode]) for mode in args.modes}

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": all_records}, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    if "sequential" in summary and "parallel" in summary:
        seq_lat = summary["sequential"]["latency_ms_mean"]
        par_lat = summary["parallel"]["latency_ms_mean"]
        if par_lat > 0:
            print(f"\nSpeedup parallel vs sequential: {seq_lat / par_lat:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
