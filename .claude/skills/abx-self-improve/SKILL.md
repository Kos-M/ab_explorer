---
name: abx-self-improve
description: Use abx to improve abx's own internal prompts — the judge prompt (EVAL_SYSTEM_PROMPT) via `abx self-optimize` / `self-eval-compare`, or the GA prompts (INITIAL_GENERATION_PROMPT, MUTATION_PROMPT, CROSSOVER_PROMPT, TEST_GENERATION_PROMPT) by running abx on them — and then promote a winner into the code safely. Use when the user says self-improve, self-optimize, meta-optimize, tune the evaluator/judge, or replace an internal prompt with an optimized one.
---

# abx self-improvement

abx's own behaviour is driven by prompt constants. Improving them changes every future experiment, so the bar for promoting a winner is higher than for a normal run.

| Constant | File | `.format()` placeholders that must survive |
|---|---|---|
| `EVAL_SYSTEM_PROMPT` | `abx/evaluator.py` | none |
| `EVAL_USER_PROMPT` | `abx/evaluator.py` | `task_description`, `response`, `rubric` |
| `INITIAL_GENERATION_PROMPT` | `abx/population.py` | `count`, `task_description` |
| `MUTATION_PROMPT` | `abx/population.py` | `task_description`, `system_prompt`, `user_prompt` |
| `CROSSOVER_PROMPT` | `abx/population.py` | `task_description`, `sys_a`, `sys_b`, `user_a`, `user_b` |
| `TEST_GENERATION_PROMPT` | `abx/test_generator.py` | `count`, `task_description`, `system_prompt`, `user_prompt` |

Literal braces in these strings (e.g. JSON examples) must be doubled `{{ }}` or `.format()` raises.

## Path A: the judge prompt (built in)

`abx self-optimize` evolves `EVAL_SYSTEM_PROMPT` against a suite with ground-truth scores.

1. **Build a calibration suite.** Different meaning from a normal suite: here `input` is an *AI response to be judged*, `rubric` is the criteria, and `expected_score` is the score a careful human would give (0–10). `generate-tests` does not produce `expected_score`, so write these by hand or have the user label them. Spread expected scores across the whole range (several low, mid and high), 15+ cases.
   - Gotcha: `expected_score: 0` is treated as 5.0 (`expected_score or 5.0` in `abx/self_optimizer.py`). Use `0.1` for "worthless" until that is fixed, and tell the user.
2. **Run** (billed; confirm with the user first):
   ```bash
   abx self-optimize --tests calib.json --cycles 5 --population 8 \
     --accuracy-weight 0.6 --cost-weight 0.2 --latency-weight 0.2
   ```
   Accuracy per case = 1 − |judge score − expected| / 10.
3. **Compare against the current prompt:**
   ```bash
   abx self-eval-compare -e <id>
   ```
   Caveat: this re-scores the baseline fresh but reuses the winner's stored scores, so the comparison is asymmetric. For a decision, re-run the compare (or both prompts) more than once and look at the spread.

## Path B: the GA / test-generation prompts (abx on itself)

There is no dedicated command; treat the internal prompt as an ordinary prompt under test.

1. Write a suite where each `input` is a realistic set of values for that prompt's placeholders (e.g. for `MUTATION_PROMPT`: a task description plus a mediocre system/user pair) and each `rubric` says what a good output looks like (valid JSON with the right keys, a genuinely better prompt, etc.).
2. `abx init --task "<what the prompt does>" --tests meta_tests.json --system-prompt <file with the current constant>` so the current version is the generation-0 baseline.
3. `abx run -e <id> ...`, then `abx report -e <id> --winner-only`.

## Promoting a winner into the code

Only promote when the winner beats the current prompt by more than run-to-run noise (Experiment A's +4% accuracy on 15 cases was marginal). Then:

1. Replace the constant's text, keeping every placeholder in the table above and doubling literal braces.
2. Check it formats: `python -c "from abx.population import MUTATION_PROMPT as P; P.format(task_description='t', system_prompt='s', user_prompt='u')"` (adapt per constant).
3. `pytest -q`.
4. Commit in the repo's style, with the evidence in the body, e.g. `feat: replace MUTATION_PROMPT with GA-optimized winner (score X, +Y% vs baseline)`, including experiment ID, suite size, generations, and baseline vs winner numbers. Only commit when the user asks.
5. Save the experiment summary next to the change (as `exp_a_run_report.md` did) so the claim can be checked later.
