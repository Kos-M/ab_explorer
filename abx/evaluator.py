"""Rubric-based LLM evaluation for ab_explorer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .models import Candidate, PromptPair, TestCase, TestSuite


EVAL_SYSTEM_PROMPT = """You are a holistic evaluator scoring AI responses against a rubric.

Focus on completeness—does the response cover all expected aspects of the task?
Use general guidelines: a score of 10 means everything is covered and well-explained;
5 means half the points are addressed; 0 means nothing is covered.

Provide a brief analysis (2-3 sentences) without mandatory chain-of-thought.
Be lenient on minor inaccuracies if the response is otherwise complete.
Emphasize coverage of key topics over perfect wording."""

EVAL_USER_PROMPT = """Evaluate the response using a three-step process:

Step 1 - Identify which rubric criteria are met, partially met, or not met.
Step 2 - Assign an integer score 0-10 (0=none met, 10=all perfectly met). Write the score before reasoning.
Step 3 - Provide a brief reasoning that cites specific elements from the response and rubric.

Step 4 (self-consistency): Ensure the reasoning supports the score. If there is a mismatch, revise either the score or reasoning until consistent.

Task: {task_description}
Response: {response}
Rubric: {rubric}

Output a JSON object with keys: "score", "reasoning", "consistent". Do not include any extra text."""


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate on one test case."""
    score: float
    reasoning: str
    output: str
    cost: float
    latency: float
    token_count: int


def evaluate_test_case(
    llm_client,
    task_description: str,
    test_case: TestCase,
    prompts: PromptPair,
) -> EvaluationResult:
    """Evaluate a candidate's prompts against a single test case.

    Steps:
    1. Run the candidate's prompts on the test case input
    2. Score the output against the rubric using the LLM
    """
    # Step 1: Get candidate's response to the input
    response = llm_client.chat(
        system_prompt=prompts.system_prompt,
        user_prompt=f"{prompts.user_prompt}\n\nInput: {test_case.input}",
        temperature=0.3,
    )
    output = response.content

    # Step 2: Score the output against the rubric
    score_response = llm_client.chat(
        system_prompt=EVAL_SYSTEM_PROMPT,
        user_prompt=EVAL_USER_PROMPT.format(
            task_description=task_description,
            response=output,
            rubric=test_case.rubric,
        ),
        temperature=0.2,
    )

    # Parse score from evaluator response
    score = 5.0  # Default mid-score
    try:
        parsed = json.loads(score_response.content)
        if isinstance(parsed, dict):
            score = float(parsed.get("score", 5))
        elif isinstance(parsed, (int, float)):
            score = float(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score = max(0.0, min(10.0, score))  # Clamp 0-10

    return EvaluationResult(
        score=score,
        reasoning=score_response.content[:200],
        output=output,
        cost=response.cost + score_response.cost,
        latency=response.latency_ms + score_response.latency_ms,
        token_count=response.total_tokens + score_response.total_tokens,
    )


def evaluate_candidate(
    llm_client,
    candidate: Candidate,
    test_suite: TestSuite,
) -> Candidate:
    """Evaluate a candidate against all test cases in a suite.

    Returns the candidate with scores populated.
    """
    total_cost = 0.0
    total_latency = 0.0
    total_tokens = 0
    scores = []

    for i, test_case in enumerate(test_suite.test_cases):
        result = evaluate_test_case(
            llm_client,
            test_suite.task_description,
            test_case,
            candidate.prompts,
        )
        scores.append(result.score)
        total_cost += result.cost
        total_latency += result.latency
        total_tokens += result.token_count

    candidate.scores = scores
    candidate.cost = total_cost
    candidate.latency = total_latency / len(test_suite.test_cases) if test_suite.test_cases else 0.0
    candidate.token_count = total_tokens

    return candidate
