"""Reference baselines for the multi-agent supervisor benchmark.

- single_agent: one LLM call with all docs concatenated. No decomposition.
- sequential: same specialist chain as the supervisor, but specialists run
  one-after-another instead of parallel via Send. Isolates the speedup that
  comes from parallelism alone (not from having more agents).
"""
