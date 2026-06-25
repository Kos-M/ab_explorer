# Experiment A: EVAL_SYSTEM_PROMPT Optimization via GA

## Config
- Generations: 5 (completed: 5 — gen 0 to 4, gen 5 timed out)
- Population: 8
- Test cases: 15 (exp_a_tests.json with expected_score)
- KPI weights: accuracy=0.6, cost=0.2, latency=0.2
- Model: deepseek-v4-flash

## Summary
- Status: completed
- Generations completed: 4
- Total cost: $0.110151
- Total tokens: 522,353
- Total candidates: 40
- Avg cost/call: $0.002754
- Avg latency: 4793ms

## Generation Winners
| Gen | Composite | Accuracy | Cost | Latency |
|-----|-----------|----------|------|---------|
| 1 | 0.6287 | 82.67% | $0.002329 | 4396ms |
| 2 | 0.5211 | 81.33% | $0.002651 | 4219ms |
| 3 | 0.5873 | 76.67% | $0.002285 | 4337ms |
| 4 | 0.6244 | 80.67% | $0.002727 | 4447ms |

## Best Candidate
- Generation: 1
- Composite score: 0.6287
- Accuracy: 82.67%
- Cost: $0.002329
- Latency: 4396ms

### Winning EVAL_SYSTEM_PROMPT

```
You are a holistic evaluator scoring AI responses against a rubric. Focus on completeness—does the response cover all expected aspects of the task? Use general guidelines: a score of 10 means everything is covered and well-explained; 5 means half the points are addressed; 0 means nothing is covered. Provide a brief analysis (2-3 sentences) without mandatory chain-of-thought. Be lenient on minor inaccuracies if the response is otherwise complete. Emphasize coverage of key topics over perfect wording.
```

### Baseline EVAL_SYSTEM_PROMPT

```
You are an expert evaluator scoring AI responses against a rubric.

Given a task, an AI's response, and a rubric, score the response from 0-10 where:
- 0: Completely fails to meet the rubric
- 5: Partially meets the rubric
- 10: Perfectly meets the rubric

Be strict but fair. Consider accuracy, completeness, and relevance.
```

## Comparison: Optimized vs Baseline
- Accuracy: +4.00% (78.67% baseline → 82.67% optimized)
- Cost: +$0.000019 ($0.002310 → $0.002329)
- Latency: -227ms (4624ms → 4396ms)

**Verdict: Marginal improvement. Accuracy improved by 4%, cost essentially unchanged, latency slightly better.**
