# multi-agent-supervisor

> LangGraph supervisor that fans out specialist agents in parallel via Send. **1.99x faster than the sequential multi-agent baseline** on a 3-sub-question workload, reproducible in 6 seconds with no API key.

[![ci](https://github.com/Tajaddin/multi-agent-supervisor/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/multi-agent-supervisor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen)](#testing)

## Hero metrics

Reproducible in 6 seconds: `python -m benchmarks.smoke_parallel --delay 0.5 --repeats 3`

| Workload | Sequential multi-agent | Parallel supervisor | Speedup |
|---|---:|---:|---:|
| 3 sub-questions, 0.5s per LLM call (Haiku-4.5-shaped) | **4.00s** | **2.01s** | **1.99x** |
| 8 LLM calls per query (1 planner + 3 analyzers + 3 verifiers + 1 synth) | identical work, serial | identical work, fanned out | same correctness, half the wall-clock |

Why this is real, not handwaved: the supervisor and the sequential baseline call **identical specialist functions on identical state**. The only difference is the graph topology. The 58-test suite includes `test_supervisor_faster_than_sequential_under_sleep`, which fails CI if the Send fan-out stops actually concurrent.

The math: 8 calls × 0.5s = 4.00s serial. Parallel collapses to `planner + max(analyzers) + max(verifiers) + synth` = 4 × 0.5s = 2.00s. We measure 2.01s — overhead under 1%.

## Architecture

```
                              START
                                |
                                v
                          +-----------+
                          |  planner  |   1 call: decompose query
                          +-----------+
                                |
                  (Send fan-out, parallel)
              +-----------------+-----------------+
              v                 v                 v
        +-----------+     +-----------+     +-----------+
        | retrieve  |     | retrieve  |     | retrieve  |    N parallel
        |  (sq-01)  |     |  (sq-02)  |     |  (sq-03)  |    Wikipedia hits
        +-----------+     +-----------+     +-----------+
              \                 |                 /
               +----------------+----------------+
                                | (barrier: merge_dicts reducer)
                                v
                        +-----------------+
                        | dispatch_analy. |
                        +-----------------+
                                |
                  (Send fan-out, parallel)
              +-----------------+-----------------+
              v                 v                 v
        +-----------+     +-----------+     +-----------+
        |  analyze  |     |  analyze  |     |  analyze  |    N parallel
        |  (sq-01)  |     |  (sq-02)  |     |  (sq-03)  |    Anthropic calls
        +-----------+     +-----------+     +-----------+
              \                 |                 /
               +----------------+----------------+
                                | (barrier)
                                v
                        +-----------------+
                        | dispatch_verify |
                        +-----------------+
                                |
                  (Send fan-out, parallel)
              +-----------------+-----------------+
              v                 v                 v
        +-----------+     +-----------+     +-----------+
        |  verify   |     |  verify   |     |  verify   |    N parallel
        |  (sq-01)  |     |  (sq-02)  |     |  (sq-03)  |    fact-checks
        +-----------+     +-----------+     +-----------+
              \                 |                 /
               +----------------+----------------+
                                | (barrier)
                                v
                          +-----------+
                          | synthesize|   1 call: merge, drop flagged claims
                          +-----------+
                                |
                                v
                               END
```

Five specialist roles. Each runs as its own LangGraph node. Three of them (retrieve, analyze, verify) are dispatched once per sub-question via `Send`, then converge on a barrier where the `merge_dicts` reducer combines per-sub-question writes without clobbering.

## Why this matters for production

JD signal this maps to:
- **Multi-agent orchestration** (Cohere Agent Infrastructure, Sekai, Mango Languages, Moore, IDC, Pair Team, n8n)
- **Agentic systems** in the broad sense (most "AI Engineer" roles in 2026)
- **Anthropic SDK production usage** (Caylent, M3 USA, ERP Suites, Pair Team)
- **LangGraph + Send + parallel state** (the framework primitive that makes this work)

The single-agent RAG pattern is table stakes in 2026. Multi-agent with parallel fan-out, shared state, per-agent verification, and observable telemetry is the next bar.

## Quick start

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
mas "Who directed the 2010 film Inception, and what film did they release in 2023?"
```

Output (truncated):

```
Final answer:
Christopher Nolan directed the 2010 film Inception. He released
Oppenheimer in July 2023, which won the 2024 Academy Award for Best Picture.

Citations:
  - Inception (https://en.wikipedia.org/wiki/Inception)
  - Christopher Nolan (https://en.wikipedia.org/wiki/Christopher_Nolan)
  - Oppenheimer (film) (https://en.wikipedia.org/wiki/Oppenheimer_(film))

Sub-questions:
  - Who directed Inception (2010)?
  - What film did Christopher Nolan release in 2023?

Elapsed: 6.18s
```

## Reproducible smoke benchmark (no API key)

```bash
python -m benchmarks.smoke_parallel --delay 0.5 --repeats 3
```

```
  [seq 1/3] 4.00s (8 LLM calls)
  [seq 2/3] 4.00s (8 LLM calls)
  [seq 3/3] 4.00s (8 LLM calls)
  [par 1/3] 2.01s (8 LLM calls)
  [par 2/3] 2.01s (8 LLM calls)
  [par 3/3] 2.01s (8 LLM calls)

Speedup parallel vs sequential: 1.99x
```

The smoke benchmark uses a `SleepyClient` (drop-in `LLMClient`) that sleeps for a fixed delay per call. It is the deterministic version of the HotpotQA benchmark below.

## HotpotQA benchmark (needs ANTHROPIC_API_KEY)

```bash
pip install -e ".[eval]"
python -m benchmarks.hotpotqa_eval --n 30 --modes single sequential parallel
```

Captures F1, EM, contains-gold, p50 / mean / max latency per mode. Writes `benchmarks/results/hotpotqa_results.json` with per-question records and a top-level summary. See [RESULTS.md](RESULTS.md) for the format and how to interpret the numbers.

## What each specialist does

| Specialist | Purpose | LLM? | Reads | Writes |
|---|---|---|---|---|
| planner | Decompose user query into 2-4 atomic sub-questions | yes (1 call) | `query` | `sub_questions` |
| retriever | Pull top-K Wikipedia snippets per sub-question | no | sub_question | `retrievals[sq.id]` |
| analyzer | Synthesize sub-answer from snippets, emit citations | yes (N calls) | sub_question + docs | `analyses[sq.id]` |
| verifier | Fact-check the analyzer's claims against snippets | yes (N calls) | sub_question + docs + analysis | `verifications[sq.id]` |
| synthesizer | Merge sub-answers, drop unsupported claims | yes (1 call) | everything | `final_answer`, `citations` |

Total LLM calls per query: `1 + 2N + 1` where N = number of sub-questions. With N=3: 8 LLM calls, parallel wall-clock = 4 stages (planner -> analyzers || -> verifiers || -> synth).

## State shape and reducers

The whole graph shares one `TypedDict`. Per-sub-question fields use the `merge_dicts` reducer so parallel writes converge without overwriting:

```python
class SupervisorState(TypedDict, total=False):
    query: str
    sub_questions: list[SubQuestion]

    retrievals:    Annotated[dict[str, list[RetrievedDoc]], merge_dicts]
    analyses:      Annotated[dict[str, Analysis],           merge_dicts]
    verifications: Annotated[dict[str, Verification],       merge_dicts]

    final_answer: str
    citations:    list[Citation]
    telemetry:    Annotated[list[TelemetryEvent], append_list]
```

If the reducer were `=` instead of `merge_dicts`, parallel retrievers would clobber each other and only the last writer's snippets would survive. `test_retrievals_keyed_by_sub_question_id` is the canary for that bug.

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=multi_agent_supervisor --cov-report=term-missing
```

58 tests, 75% coverage. The key tests:

- `test_state.py::TestMergeDicts::test_simulates_parallel_specialist_writes` — proves the reducer is order-independent
- `test_supervisor.py::TestSupervisorEndToEnd::test_retrievals_keyed_by_sub_question_id` — proves the graph does not clobber parallel writes
- `test_supervisor.py::TestSupervisorEndToEnd::test_llm_invoked_for_every_specialist_layer` — pins the LLM-call budget at 1 + 2N + 1
- `test_baselines.py::TestParallelismActuallyParallel::test_supervisor_faster_than_sequential_under_sleep` — fails if Send stops being concurrent

## Docker

```bash
docker build -t multi-agent-supervisor:latest .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY multi-agent-supervisor:latest \
  "Which country was first to ratify the Kyoto Protocol?"
```

`docker compose up smoke` runs the no-API smoke benchmark.

## Project layout

```
src/multi_agent_supervisor/
  state.py             # SupervisorState + reducers
  supervisor.py        # LangGraph topology (Send fan-outs + barriers)
  llm.py               # LLMClient protocol + AnthropicClient + MockClient
  cli.py               # `mas` entry point
  agents/
    planner.py
    retriever.py
    analyzer.py
    verifier.py
    synthesizer.py
  tools/
    wikipedia.py       # WikipediaSearcher + StaticSearcher (for tests)

benchmarks/
  smoke_parallel.py    # no-API speedup benchmark (6 sec)
  hotpotqa_eval.py     # real HotpotQA F1/EM/latency benchmark
  baselines/
    single_agent.py    # one LLM call, no decomposition
    sequential.py      # specialists run serially (isolates parallelism gain)
  results/
    smoke_parallel.json
    hotpotqa_results.json
```

## License

MIT
