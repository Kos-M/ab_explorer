"""Tests for abx.llm."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from abx.llm import DeepSeekClient, LLMResponse, _calculate_cost


class TestCalculateCost:
    def test_zero_tokens(self):
        cost = _calculate_cost(0, 0)
        assert cost == 0.0

    def test_small_usage(self):
        cost = _calculate_cost(100, 50)
        assert cost > 0
        assert cost < 0.001  # Should be ~$0.000082

    def test_large_usage(self):
        cost = _calculate_cost(1_000_000, 500_000)
        assert round(cost, 2) == 0.82  # $0.27 + $0.55 = $0.82


class TestDeepSeekClientInit:
    def test_no_key_raises(self):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient(api_key="")

    def test_key_from_arg(self):
        client = DeepSeekClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "deepseek-chat"
        assert client.base_url == "https://api.deepseek.com"

    def test_custom_config(self):
        client = DeepSeekClient(
            api_key="key",
            base_url="https://custom.example.com",
            model="deepseek-v4-flash",
            timeout=30.0,
        )
        assert client.base_url == "https://custom.example.com"
        assert client.model == "deepseek-v4-flash"
        assert client.timeout == 30.0


class TestDeepSeekClientChat:
    @patch("abx.llm.httpx.Client")
    def test_basic_chat(self, mock_client_class):
        mock_instance = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_instance

        # Mock the API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        mock_instance.post.return_value = mock_response

        client = DeepSeekClient(api_key="test-key")
        result = client.chat(
            system_prompt="Be helpful",
            user_prompt="Hello",
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello! How can I help?"
        assert result.model == "deepseek-chat"
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10
        assert result.total_tokens == 30
        assert result.latency_ms > 0
        assert result.cost > 0

        # Verify the request was built correctly
        call_args = mock_instance.post.call_args
        assert call_args is not None
        url = call_args[0][0]
        payload = call_args[1]["json"]
        assert "chat/completions" in url
        assert payload["model"] == "deepseek-chat"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @patch("abx.llm.httpx.Client")
    def test_chat_no_system_prompt(self, mock_client_class):
        mock_instance = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        mock_instance.post.return_value = mock_response

        client = DeepSeekClient(api_key="test-key")
        result = client.chat(user_prompt="Hello")

        assert result.content == "Hi"

        # Verify only 1 message (no system prompt)
        call_args = mock_instance.post.call_args
        payload = call_args[1]["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @patch("abx.llm.httpx.Client")
    def test_chat_api_error(self, mock_client_class):
        mock_instance = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=MagicMock()
        )
        mock_instance.post.return_value = mock_response

        client = DeepSeekClient(api_key="bad-key")
        with pytest.raises(httpx.HTTPStatusError):
            client.chat(user_prompt="Hello")
