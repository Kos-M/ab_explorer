"""Tests for abx.cli."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from abx.cli import app

runner = CliRunner()


def test_init_missing_file():
    result = runner.invoke(app, ["init", "--task", "test", "--tests", "nonexistent.json"])
    assert result.exit_code != 0
    assert "not found" in result.stdout


def test_init_with_valid_file():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "task_description": "Extract dates",
        "evaluation_model": "deepseek-v4-flash",
        "test_cases": [
            {"input": "March 5, 2024", "rubric": "Must extract date"},
        ],
    }, tmp)
    tmp.close()

    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()

    result = runner.invoke(app, [
        "init",
        "--task", "Extract dates from text",
        "--tests", tmp.name,
        "--output", db.name,
    ])
    assert result.exit_code == 0
    assert "Experiment initialized" in result.stdout

    os.unlink(tmp.name)
    os.unlink(db.name)


def test_init_empty_test_cases():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "task_description": "Test",
        "evaluation_model": "deepseek-v4-flash",
        "test_cases": [],
    }, tmp)
    tmp.close()

    result = runner.invoke(app, [
        "init",
        "--task", "test",
        "--tests", tmp.name,
    ])
    assert result.exit_code != 0

    os.unlink(tmp.name)


def test_run_missing_experiment():
    result = runner.invoke(app, [
        "run", "--experiment-id", "nonexistent",
    ])
    assert result.exit_code != 0
    assert "not found" in result.stdout


def test_list_experiments_empty():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    result = runner.invoke(app, [
        "list-experiments", "--db", db.name,
    ])
    assert result.exit_code == 0
    assert "No experiments found" in result.stdout
    os.unlink(db.name)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "run" in result.stdout
    assert "report" in result.stdout
    assert "list-experiments" in result.stdout
    assert "generate-tests" in result.stdout


# --- generate-tests command tests ---

def test_generate_tests_missing_api_key():
    """Test that generate-tests shows error when DEEPSEEK_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(app, [
            "generate-tests",
            "--system-prompt", "You are a helpful assistant.",
            "--user-prompt", "Answer the question.",
            "--task", "General QA",
        ])
        assert result.exit_code != 0
        assert "DEEPSEEK_API_KEY" in result.stdout


def test_generate_tests_empty_prompts():
    """Test that empty system+user prompts produce error."""
    result = runner.invoke(app, [
        "generate-tests",
        "--system-prompt", "",
        "--user-prompt", "",
        "--task", "Test",
    ])
    assert result.exit_code != 0
    assert "non-empty" in result.stdout


def test_generate_tests_reads_from_files():
    """Test that prompts can be read from files."""
    sys_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    sys_file.write("You are a math tutor.")
    sys_file.close()

    user_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    user_file.write("Solve this math problem.")
    user_file.close()

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
        with patch("abx.cli.DeepSeekClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.chat.return_value.content = (
                '{"test_cases": [{"input": "2+2", "rubric": "Must be correct"}]}'
            )
            out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            out_file.close()

            result = runner.invoke(app, [
                "generate-tests",
                "--system-prompt", sys_file.name,
                "--user-prompt", user_file.name,
                "--task", "Math tutoring",
                "--output", out_file.name,
                "--count", "1",
            ])

            assert result.exit_code == 0
            assert "1 test cases" in result.stdout

            # Verify output file was written
            with open(out_file.name) as f:
                data = json.load(f)
                assert data["task_description"] == "Math tutoring"
                assert len(data["test_cases"]) == 1
                assert data["test_cases"][0]["input"] == "2+2"

            os.unlink(out_file.name)

    os.unlink(sys_file.name)
    os.unlink(user_file.name)


def test_generate_tests_inline_prompts():
    """Test generate-tests with inline prompt strings (not files)."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
        with patch("abx.cli.DeepSeekClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.chat.return_value.content = (
                '{"test_cases": [{"input": "Hi", "rubric": "Must be polite"}]}'
            )
            out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            out_file.close()

            result = runner.invoke(app, [
                "generate-tests",
                "--system-prompt", "Be polite and helpful.",
                "--user-prompt", "Respond to the user.",
                "--task", "Polite assistant",
                "--output", out_file.name,
                "--count", "1",
            ])

            assert result.exit_code == 0
            assert "Generated" in result.stdout

            with open(out_file.name) as f:
                data = json.load(f)
                assert len(data["test_cases"]) == 1

            os.unlink(out_file.name)


def test_generate_tests_count_validation():
    """Test that count must be >= 1."""
    result = runner.invoke(app, [
        "generate-tests",
        "--system-prompt", "Be helpful.",
        "--user-prompt", "Answer.",
        "--task", "Test",
        "--count", "0",
    ])
    assert result.exit_code != 0


def test_generate_tests_with_custom_model():
    """Test that custom model is passed through."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
        with patch("abx.cli.DeepSeekClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.chat.return_value.content = (
                '{"test_cases": [{"input": "Hello", "rubric": "Must respond"}]}'
            )
            out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            out_file.close()

            result = runner.invoke(app, [
                "generate-tests",
                "--system-prompt", "Be helpful.",
                "--user-prompt", "Answer the user.",
                "--task", "General chat",
                "--output", out_file.name,
                "--model", "deepseek-v4-flash",
                "--count", "1",
            ])

            assert result.exit_code == 0
            # Verify DeepSeekClient was called with custom model
            assert MockClient.call_args[1]["model"] == "deepseek-v4-flash"

            os.unlink(out_file.name)
