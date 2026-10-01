---
name: abx-run-experiment
description: Run a prompt-optimization experiment end to end with abx — init from a tests.json, run the genetic algorithm, then read results with report/list-experiments and extract the winning prompt. Use when the user wants to optimize, A/B test, or evolve a prompt, run or resume an experiment, check experiment status, or see which prompt won.
---

# abx experiments

Flow: `init` (store suite + config) → `run` (GA loop, billed) → `report` (read results). All state lives in SQLite, default `ab_explorer.db` in the current directory; every command takes `--db` / `-d` (for `init` it is `--output` / `-o`).

Prerequisites: a working install and a working provider — logged-in `claude` CLI (default) or `DEEPSEEK_API_KEY` with `--provider deepseek` (`abx-setup`), and a reviewed tests.json (`abx-generate-tests`).

## 1. Init

```bash
abx init --task "Extract calendar dates from text" --tests tests.json \
  --name "Date extraction v1" \
  --system-prompt prompts/system.txt     # optional: current prompt, file path or inline text
```

Passing the current system prompt seeds it into generation 0 as the baseline candidate, so the GA improves on it instead of starting blind. Do this whenever the user already has a prompt. Note the printed experiment ID.

## 2. Estimate cost, then confirm

Calls per run ≈ population × (cycles + 1) × test cases × 2, plus a few generation calls. Example: 5 × 21 × 10 × 2 ≈ 2,100 calls. With `--provider claude` each call is a separate `claude -p` process (roughly 1–3 s, fractions of a cent on haiku), so a 2,000-call run takes on the order of an hour. With `--provider deepseek`, Experiment A (pop 8, 15 cases, 5 gens) cost about $0.11 and 522k tokens on deepseek-v4-flash. State the estimate and get the user's go-ahead before a large run.

## 3. Run

```bash
abx run -e <id> --cycles 20 --population 5 \
  --accuracy-weight 0.5 --cost-weight 0.3 --latency-weight 0.2 \
  --plateau-threshold 0.02 --plateau-rounds 5
```

- Runs are long and sequential. Run it in the background and check on it rather than blocking.
- `run` always starts again from generation 0; there is no resume. CLI flags overwrite the stored config.
- Stops early when the best score varies ≤ `plateau-threshold` (absolute, not %) over the last `plateau-rounds` generations.
- If the user cares mostly about quality, raise `--accuracy-weight` (e.g. 0.8/0.1/0.1).

## 4. Read results

```bash
abx list-experiments                 # id, name, status, generation
abx report -e <id>                   # stats + best score per generation
abx report -e <id> --winner-only     # full winning system + user prompt
```

## Interpreting results honestly

- Cost and latency are scored relative to the slowest/most expensive candidate **in the same generation**, so composite scores are not strictly comparable across generations. Compare accuracy (avg rubric score) too.
- Scores come from an LLM judge with a 5.0 fallback when its JSON can't be parsed (`abx/evaluator.py`), so a cluster of exact 5.0s means parse failures, not mediocre prompts.
- With ~10 test cases, differences of a few points are within noise. Say so instead of declaring a winner.

When handing back a winner, show the prompt, its score and accuracy versus generation 0 (the baseline when seeded), and the run's total cost.
