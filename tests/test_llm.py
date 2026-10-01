"""Tests for abx.llm."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from abx.llm import ClaudeCLIClient, DeepSeekClient, LLMResponse, _calculate_cost, make_client


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


CLI_OK = json.dumps({
    "type": "result",
    "is_error": False,
    "result": "Hello! How can I help?",
    "duration_ms": 978,
    "total_cost_usd": 0.0016,
    "usage": {
        "input_tokens": 20,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 0,
        "output_tokens": 10,
    },
    "modelUsage": {"claude-haiku-4-5-20251001": {}},
})


class TestClaudeCLIClientInit:
    @patch("abx.llm.shutil.which", return_value=None)
    def test_missing_cli_raises(self, _which, monkeypatch):
        monkeypatch.delenv("ABX_CLAUDE_CLI", raising=False)
        with pytest.raises(ValueError, match="claude CLI not found"):
            ClaudeCLIClient()

    @patch("abx.llm.shutil.which", return_value="/usr/bin/claude")
    def test_defaults(self, _which, monkeypatch):
        monkeypatch.delenv("ABX_CLAUDE_CLI", raising=False)
        monkeypatch.delenv("ABX_CLAUDE_MODEL", raising=False)
        client = ClaudeCLIClient()
        assert client.cli_path == "/usr/bin/claude"
        assert client.model == "haiku"

    def test_custom_config(self):
        client = ClaudeCLIClient(model="sonnet", timeout=30.0, cli_path="/opt/claude")
        assert client.model == "sonnet"
        assert client.timeout == 30.0
        assert client.cli_path == "/opt/claude"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("ABX_CLAUDE_MODEL", "opus")
        client = ClaudeCLIClient(cli_path="/opt/claude")
        assert client.model == "opus"


class TestClaudeCLIClientChat:
    @patch("abx.llm.subprocess.run")
    def test_basic_chat(self, mock_run):
        captured = {}

        def fake_run(cmd, **kwargs):
            idx = cmd.index("--system-prompt-file")
            with open(cmd[idx + 1], encoding="utf-8") as f:
                captured["system"] = f.read()
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _completed(CLI_OK)

        mock_run.side_effect = fake_run

        client = ClaudeCLIClient(cli_path="/opt/claude", model="haiku")
        result = client.chat(system_prompt="Be helpful", user_prompt="Hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello! How can I help?"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.prompt_tokens == 25
        assert result.completion_tokens == 10
        assert result.total_tokens == 35
        assert result.latency_ms == 978
        assert result.cost == 0.0016

        cmd = captured["cmd"]
        assert cmd[0] == "/opt/claude"
        assert "-p" in cmd
        assert cmd[cmd.index("--model") + 1] == "haiku"
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--tools") + 1] == ""
        assert captured["system"] == "Be helpful"
        assert captured["kwargs"]["input"] == "Hello"

    @patch("abx.llm.subprocess.run")
    def test_chat_no_system_prompt(self, mock_run):
        mock_run.return_value = _completed(CLI_OK)
        client = ClaudeCLIClient(cli_path="/opt/claude")
        client.chat(user_prompt="Hello")

        cmd = mock_run.call_args[0][0]
        assert "--system-prompt-file" not in cmd

    @patch("abx.llm.subprocess.run")
    def test_chat_cli_error_result(self, mock_run):
        mock_run.return_value = _completed(
            json.dumps({"is_error": True, "result": "Not logged in"}), returncode=1
        )
        client = ClaudeCLIClient(cli_path="/opt/claude")
        with pytest.raises(RuntimeError, match="Not logged in"):
            client.chat(user_prompt="Hello")

    @patch("abx.llm.subprocess.run")
    def test_chat_non_json_output(self, mock_run):
        mock_run.return_value = _completed("", returncode=2, stderr="unknown option")
        client = ClaudeCLIClient(cli_path="/opt/claude")
        with pytest.raises(RuntimeError, match="unknown option"):
            client.chat(user_prompt="Hello")


# --- DeepSeek ---------------------------------------------------------------

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
        assert round(cost, 2) == 0.28  # $0.14 + $0.14 = $0.28


class TestDeepSeekClientInit:
    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient(api_key="")

    def test_key_from_arg(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        client = DeepSeekClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "deepseek-v4-flash"
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
            "model": "deepseek-v4-flash",
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
        assert result.model == "deepseek-v4-flash"
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
        assert payload["model"] == "deepseek-v4-flash"
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
            "model": "deepseek-v4-flash",
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


    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.com")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = DeepSeekClient(api_key="key")
        assert client.base_url == "https://env.example.com"
        assert client.model == "deepseek-chat"


# --- Provider selection -----------------------------------------------------

class TestMakeClient:
    def test_default_is_claude(self, monkeypatch):
        monkeypatch.delenv("ABX_PROVIDER", raising=False)
        monkeypatch.setenv("ABX_CLAUDE_CLI", "/opt/claude")
        assert isinstance(make_client(), ClaudeCLIClient)

    def test_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        client = make_client("deepseek")
        assert isinstance(client, DeepSeekClient)
        assert client.model == "deepseek-v4-flash"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("ABX_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
        assert isinstance(make_client(), DeepSeekClient)

    def test_model_passed_through(self, monkeypatch):
        monkeypatch.setenv("ABX_CLAUDE_CLI", "/opt/claude")
        assert make_client("claude", "sonnet").model == "sonnet"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            make_client("gpt")
