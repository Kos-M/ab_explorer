"""Tests for abx.test_generator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from abx.llm import LLMResponse
from abx.test_generator import (
    TEST_GENERATION_PROMPT,
    _parse_json_response,
    _strip_code_fence,
    generate_test_suite,
)


class TestStripCodeFence:
    def test_no_fence(self):
        assert _strip_code_fence("plain text") == "plain text"

    def test_triple_backtick_fence(self):
        text = "```\n{\"key\": \"value\"}\n```"
        assert _strip_code_fence(text) == "{\"key\": \"value\"}"

    def test_json_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert _strip_code_fence(text) == "{\"key\": \"value\"}"

    def test_only_opening_fence(self):
        text = "```\ncontent"
        assert _strip_code_fence(text) == "content"

    def test_only_closing_fence(self):
        text = "content\n```"
        assert _strip_code_fence(text) == "content"


class TestParseJsonResponse:
    def test_direct_json(self):
        result = _parse_json_response('{"test_cases": [{"input": "a", "rubric": "b"}]}')
        assert result == {"test_cases": [{"input": "a", "rubric": "b"}]}

    def test_fenced_json(self):
        text = "```json\n{\"test_cases\": [{\"input\": \"x\", \"rubric\": \"y\"}]}\n```"
        result = _parse_json_response(text)
        assert result == {"test_cases": [{"input": "x", "rubric": "y"}]}

    def test_json_with_prefix_text(self):
        text = "Here is the result:\n{\"test_cases\": [{\"input\": \"a\", \"rubric\": \"b\"}]}"
        result = _parse_json_response(text)
        assert result == {"test_cases": [{"input": "a", "rubric": "b"}]}

    def test_invalid_json(self):
        result = _parse_json_response("not json at all")
        assert result is None

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result is None


class TestGenerateTestSuite:
    def test_successful_generation(self):
        """Test that valid LLM response produces a TestSuite."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({
                "test_cases": [
                    {"input": "What is 2+2?", "rubric": "Must answer correctly"},
                    {"input": "Explain gravity", "rubric": "Must be scientifically accurate"},
                ]
            }),
            model="deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            cost=0.0001,
        )

        suite = generate_test_suite(
            llm_client=mock_llm,
            task_description="Math and science QA",
            system_prompt="You are a helpful tutor.",
            user_prompt="Answer the question.",
            count=2,
        )

        assert suite.task_description == "Math and science QA"
        assert len(suite.test_cases) == 2
        assert suite.test_cases[0].input == "What is 2+2?"
        assert suite.test_cases[0].rubric == "Must answer correctly"
        assert suite.test_cases[1].input == "Explain gravity"
        assert suite.evaluation_model == "deepseek-v4-flash"

    def test_generation_with_count_limit(self):
        """Test that count parameter limits test cases."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({
                "test_cases": [
                    {"input": f"Input {i}", "rubric": f"Rubric {i}"}
                    for i in range(10)
                ]
            }),
            model="deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            cost=0.0001,
        )

        suite = generate_test_suite(
            llm_client=mock_llm,
            task_description="Test",
            system_prompt="System",
            user_prompt="User",
            count=3,
        )

        assert len(suite.test_cases) == 3

    def test_invalid_response_raises_error(self):
        """Test that non-JSON response raises ValueError."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="I cannot generate test cases right now.",
            model="deepseek-v4-flash",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            cost=0.00005,
        )

        with pytest.raises(ValueError, match="Failed to generate test cases"):
            generate_test_suite(
                llm_client=mock_llm,
                task_description="Test",
                system_prompt="System",
                user_prompt="User",
            )

    def test_empty_test_cases_raises_error(self):
        """Test that empty test_cases list raises ValueError."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({"test_cases": []}),
            model="deepseek-v4-flash",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            cost=0.00005,
        )

        with pytest.raises(ValueError, match="empty or invalid"):
            generate_test_suite(
                llm_client=mock_llm,
                task_description="Test",
                system_prompt="System",
                user_prompt="User",
            )

    def test_missing_test_cases_key_raises_error(self):
        """Test that response missing test_cases key raises ValueError."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({"something_else": []}),
            model="deepseek-v4-flash",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            cost=0.00005,
        )

        with pytest.raises(ValueError, match="Failed to generate test cases"):
            generate_test_suite(
                llm_client=mock_llm,
                task_description="Test",
                system_prompt="System",
                user_prompt="User",
            )

    def test_prompt_format(self):
        """Test that the generation prompt is correctly formatted."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({"test_cases": [{"input": "a", "rubric": "b"}]}),
            model="deepseek-v4-flash",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            cost=0.00005,
        )

        generate_test_suite(
            llm_client=mock_llm,
            task_description="My custom task",
            system_prompt="My system prompt",
            user_prompt="My user prompt",
            count=1,
        )

        # Check that the chat was called with the right args
        call_kwargs = mock_llm.chat.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a QA engineer. Return only valid JSON."
        assert "My custom task" in call_kwargs["user_prompt"]
        assert "My system prompt" in call_kwargs["user_prompt"]
        assert "My user prompt" in call_kwargs["user_prompt"]
        assert "1" in call_kwargs["user_prompt"]  # count

    def test_default_model(self):
        """Test that default model is used when not specified."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps({"test_cases": [{"input": "a", "rubric": "b"}]}),
            model="deepseek-v4-flash",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            cost=0.00005,
        )

        suite = generate_test_suite(
            llm_client=mock_llm,
            task_description="Test",
            system_prompt="System",
            user_prompt="User",
            count=1,
        )

        assert suite.evaluation_model == "deepseek-v4-flash"
