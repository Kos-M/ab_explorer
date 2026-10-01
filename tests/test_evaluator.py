"""Tests for abx.evaluator."""

import json
from unittest.mock import MagicMock

from abx.evaluator import EvaluationResult, evaluate_candidate, evaluate_test_case, parse_score
from abx.models import Candidate, PromptPair, TestCase, TestSuite


def make_mock_llm(run_response: str, score_response: str):
    """Create a mock LLM client. First call returns run_response, second returns score_response."""
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            resp.content = run_response
            resp.cost = 0.001
            resp.latency_ms = 200.0
            resp.total_tokens = 50
        else:
            resp.content = score_response
            resp.cost = 0.0005
            resp.latency_ms = 150.0
            resp.total_tokens = 30
        return resp

    client.chat.side_effect = side_effect
    return client


class TestEvaluateTestCase:
    def test_basic_evaluation(self):
        llm = make_mock_llm(
            run_response="The extracted date is March 5, 2024.",
            score_response=json.dumps({"score": 9, "reasoning": "Correctly extracted the date."}),
        )
        result = evaluate_test_case(
            llm,
            "Extract dates from text",
            TestCase(input="Event on March 5, 2024", rubric="Must extract the exact date"),
            PromptPair(system_prompt="Extract dates", user_prompt="Find the date"),
        )
        assert isinstance(result, EvaluationResult)
        assert result.score == 9.0
        assert result.cost > 0
        assert result.latency > 0
        assert result.token_count > 0

    def test_default_score_on_bad_response(self):
        llm = make_mock_llm(
            run_response="Hello world",
            score_response="I cannot evaluate this",
        )
        result = evaluate_test_case(
            llm,
            "Test",
            TestCase(input="hi", rubric="be helpful"),
            PromptPair(system_prompt="sys", user_prompt="user"),
        )
        # Default mid-score 5.0 when JSON parsing fails
        assert result.score == 5.0


class TestParseScore:
    def test_bare_json(self):
        assert parse_score('{"score": 7, "reasoning": "ok"}') == 7.0

    def test_json_code_fence(self):
        # The Claude CLI usually wraps JSON replies in a fence.
        assert parse_score('```json\n{\n  "score": 3,\n  "reasoning": "x"\n}\n```') == 3.0

    def test_plain_code_fence(self):
        assert parse_score('```\n{"score": 2}\n```') == 2.0

    def test_object_inside_text(self):
        assert parse_score('Here is my verdict: {"score": 8} — done.') == 8.0

    def test_bare_number(self):
        assert parse_score("6") == 6.0

    def test_zero_is_kept(self):
        assert parse_score('{"score": 0}') == 0.0

    def test_clamped(self):
        assert parse_score('{"score": 14}') == 10.0
        assert parse_score('{"score": -2}') == 0.0

    def test_default_on_garbage(self):
        assert parse_score("I cannot evaluate this") == 5.0
        assert parse_score('{"score": "high"}') == 5.0


class TestEvaluateCandidate:
    def test_evaluate_full_candidate(self):
        llm = make_mock_llm(
            run_response="Answer A",
            score_response=json.dumps({"score": 8}),
        )
        cand = Candidate(prompts=PromptPair(system_prompt="sys", user_prompt="user"))
        suite = TestSuite(
            task_description="Test task",
            test_cases=[
                TestCase(input="q1", rubric="be accurate"),
                TestCase(input="q2", rubric="be concise"),
            ],
        )
        # Need 1 call per test case eval (2 runs × 2 scores = 4 LLM calls)
        llm.chat.side_effect = None
        call_results = []
        for _ in range(4):
            r = MagicMock()
            r.content = "Answer" if _ % 2 == 0 else json.dumps({"score": 8})
            r.cost = 0.001
            r.latency_ms = 100.0
            r.total_tokens = 25
            call_results.append(r)
        llm.chat.side_effect = call_results

        result = evaluate_candidate(llm, cand, suite)
        assert len(result.scores) == 2
        assert result.cost > 0
        assert result.latency > 0
