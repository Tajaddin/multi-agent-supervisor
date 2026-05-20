# Benchmark results

Two benchmarks: a deterministic smoke benchmark (no API key, 6 seconds) and a real HotpotQA evaluation (needs ANTHROPIC_API_KEY). Both are scripted so any reviewer can reproduce them in one command.

## Smoke benchmark

Run: `python -m benchmarks.smoke_parallel --delay 0.5 --repeats 3`

Per-LLM-call delay is fixed at 0.5 seconds via a `SleepyClient` that drop-in-replaces `AnthropicClient`. Same code path otherwise. Three sub-questions per query, eight LLM calls per query.

```json
{
  "delay_per_llm_call_seconds": 0.5,
  "n_sub_questions": 3,
  "total_llm_calls_per_query": 8,
  "repeats": 3,
  "sequential_mean_seconds": 4.004,
  "parallel_mean_seconds": 2.014,
  "speedup": 1.99
}
```

Theoretical optimum: 4 sequential stages (planner -> analyzers in parallel -> verifiers in parallel -> synthesizer) at 0.5s each = 2.00s. Measured 2.01s. Overhead 0.4%.

## HotpotQA benchmark

Run with ANTHROPIC_API_KEY set:

```bash
python -m benchmarks.hotpotqa_eval --n 30 --modes single sequential parallel --seed 7
```

Dataset: HotpotQA distractor validation split, seed-7 shuffle, first 30 examples. Multi-hop QA where most questions cannot be answered from one Wikipedia article.

Per-question records and the summary block land at `benchmarks/results/hotpotqa_results.json`.

### Summary schema

```json
{
  "summary": {
    "single":     {"n": 30, "f1_mean": 0.0, "em_mean": 0.0, "contains_gold_mean": 0.0,
                   "latency_ms_mean": 0.0, "latency_ms_p50": 0.0, "latency_ms_max": 0.0},
    "sequential": { ... same fields ... },
    "parallel":   { ... same fields ... }
  },
  "records": [
    {"mode": "single", "qid": "...", "question": "...", "gold": "...", "prediction": "...",
     "f1": 0.0, "em": 0.0, "contains_gold": 0.0, "latency_ms": 0.0,
     "citations": ["..."], "tokens_in": 0, "tokens_out": 0}
  ]
}
```

### What to expect

The hero claim is "1.99x parallel vs sequential" on Haiku-4.5-shaped latency. On real Haiku 4.5, individual LLM call latencies vary, which usually pushes the parallel speedup higher than the deterministic case (the sequential baseline pays the worst-case latency at every step; parallel pays it once per layer).

F1 and EM should be approximately equal between sequential and parallel because they run the same specialists on the same state. Any gap would mean the parallel pattern lost information through the reducer.

Single-agent baseline F1 should be lower than multi-agent F1 on multi-hop questions, because the single-agent pattern feeds all docs into one prompt without per-hop verification.

### How to verify after a run

```bash
jq '.summary' benchmarks/results/hotpotqa_results.json
```

```bash
jq '.summary.sequential.latency_ms_mean / .summary.parallel.latency_ms_mean' \
   benchmarks/results/hotpotqa_results.json
```

Should print a number > 1. If it does not, parallelism is broken.
