"""Test case generator for ab_explorer.

Generates test cases (input + rubric) from existing prompts using LLM,
enabling users to create tests.json files for experiment initialization.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .models import TestCase, TestSuite


TEST_GENERATION_PROMPT = """You are a QA engineer generating test cases for evaluating AI prompt quality.

Given a task description and a set of AI prompts (system + user), generate {count} diverse test cases.

Each test case must have:
1. "input": A realistic input that a user would provide to this AI system
2. "rubric": A clear, measurable evaluation criterion (0-10 scale) for scoring the AI's response

The test cases should:
- Cover different difficulty levels (easy, medium, hard)
- Test edge cases and common scenarios
- Be diverse in input type, length, and complexity
- Have specific, actionable rubrics that a second LLM can score against

Task: {task_description}

System Prompt: {system_prompt}

User Prompt: {user_prompt}

Return ONLY a JSON object with a "test_cases" array, where each element has "input" and "rubric" keys.
No markdown, no code blocks — just raw JSON.

Example:
{{"test_cases": [
    {{"input": "Sample input 1", "rubric": "Must handle X correctly"}},
    {{"input": "Sample input 2", "rubric": "Should demonstrate Y"}}
]}}"""


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("json"):
        text = text[4:]
    return text.strip()


def _parse_json_response(text: str) -> Optional[dict]:
    """Try to parse JSON from LLM response, with fallbacks."""
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def generate_test_suite(
    llm_client,
    task_description: str,
    system_prompt: str,
    user_prompt: str,
    count: int = 5,
    model: str = "deepseek-v4-flash",
) -> TestSuite:
    """Generate a TestSuite from existing prompts using LLM.

    Args:
        llm_client: DeepSeekClient instance.
        task_description: Description of what the prompt is for.
        system_prompt: The system prompt to generate test cases from.
        user_prompt: The user prompt template.
        count: Number of test cases to generate (default: 5).
        model: Evaluation model name (default: deepseek-v4-flash).

    Returns:
        A TestSuite with generated test cases.

    Raises:
        ValueError: If test case generation fails or returns empty.
    """
    prompt = TEST_GENERATION_PROMPT.format(
        count=count,
        task_description=task_description,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    response = llm_client.chat(
        system_prompt="You are a QA engineer. Return only valid JSON.",
        user_prompt=prompt,
        temperature=0.7,
        max_tokens=4096,
    )

    parsed = _parse_json_response(response.content)
    if not parsed or "test_cases" not in parsed:
        raise ValueError(
            "Failed to generate test cases. LLM response did not contain valid test_cases JSON."
        )

    test_cases_data = parsed["test_cases"]
    if not isinstance(test_cases_data, list) or len(test_cases_data) == 0:
        raise ValueError("Generated test cases list is empty or invalid.")

    test_cases = []
    for tc in test_cases_data[:count]:
        if isinstance(tc, dict) and "input" in tc and "rubric" in tc:
            test_cases.append(TestCase(input=tc["input"], rubric=tc["rubric"]))

    if not test_cases:
        raise ValueError("No valid test cases could be parsed from LLM response.")

    return TestSuite(
        task_description=task_description,
        test_cases=test_cases,
        evaluation_model=model,
    )
