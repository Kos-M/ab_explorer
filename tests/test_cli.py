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
    assert "list-experiments" in result.stdout


def test_init_help_shows_system_prompt_option():
    """Verify --system-prompt appears in init help."""
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "--system-prompt" in result.stdout
    assert "-s" in result.stdout


def test_init_with_system_prompt_inline():
    """Inline system prompt should be stored in experiment config."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "task_description": "Translate text",
        "test_cases": [
            {"input": "Hello", "rubric": "Must translate to French"},
        ],
    }, tmp)
    tmp.close()

    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()

    result = runner.invoke(app, [
        "init",
        "--task", "Translate English to French",
        "--tests", tmp.name,
        "--output", db.name,
        "--system-prompt", "You are a professional translator.",
    ])
    assert result.exit_code == 0
    assert "System prompt" in result.stdout
    assert "professional translator" in result.stdout

    # Verify it was stored
    from abx.storage import Storage
    storage = Storage(db_path=db.name)
    experiments = storage.list_experiments()
    assert len(experiments) > 0
    exp = storage.get_experiment(experiments[0]["id"])
    assert exp is not None
    assert exp.config.system_prompt == "You are a professional translator."

    os.unlink(tmp.name)
    os.unlink(db.name)


def test_init_with_system_prompt_from_file():
    """File path as system prompt should read file content."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "task_description": "Test",
        "test_cases": [{"input": "Hi", "rubric": "Must reply"}],
    }, tmp)
    tmp.close()

    prompt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    prompt_file.write("You are a test assistant.")
    prompt_file.close()

    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()

    result = runner.invoke(app, [
        "init",
        "--task", "Test task",
        "--tests", tmp.name,
        "--output", db.name,
        "--system-prompt", prompt_file.name,
    ])
    assert result.exit_code == 0
    assert "System prompt" in result.stdout
    assert "test assistant" in result.stdout

    # Verify stored as resolved text
    from abx.storage import Storage
    storage = Storage(db_path=db.name)
    experiments = storage.list_experiments()
    exp = storage.get_experiment(experiments[0]["id"])
    assert exp is not None
    assert exp.config.system_prompt == "You are a test assistant."

    os.unlink(tmp.name)
    os.unlink(prompt_file.name)
    os.unlink(db.name)


# --- report command stats tests ---

def test_report_shows_stats():
    """Verify the report command displays experiment stats."""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()

    from abx.models import Experiment, Candidate, PromptPair
    from abx.storage import Storage

    # Create experiment with candidates
    storage = Storage(db_path=db.name)
    exp = Experiment(name="stats-report", task_description="test stats")
    storage.save_experiment(exp)

    c1 = Candidate(
        prompts=PromptPair(system_prompt="A", user_prompt="B"),
        generation=0, scores=[8.0], composite_score=8.0,
        cost=0.005, latency=150.0, token_count=500,
    )
    c2 = Candidate(
        prompts=PromptPair(system_prompt="C", user_prompt="D"),
        generation=1, scores=[9.0], composite_score=9.0,
        cost=0.003, latency=100.0, token_count=300,
    )
    storage.save_candidate(c1, exp.id)
    storage.save_candidate(c2, exp.id)
    storage.save_winner(exp.id, c2.id, rank=1, generation=1)

    result = runner.invoke(app, [
        "report", "--experiment-id", exp.id, "--db", db.name,
    ])

    assert result.exit_code == 0
    assert "Experiment Stats" in result.stdout
    assert "Total Cost" in result.stdout
    assert "Total Tokens" in result.stdout
    assert "LLM Calls" in result.stdout
    assert "Avg Cost/Call" in result.stdout
    assert "Duration" in result.stdout
    assert "Generations" in result.stdout

    os.unlink(db.name)


def test_generate_tests_long_inline_prompt_no_oserror():
    """Long inline system_prompt should NOT raise OSError in generate_tests."""
    long_prompt = "Be helpful. " * 200  # ~3000 chars — exceeds NAME_MAX on most FS
    result = runner.invoke(app, [
        "generate-tests",
        "--system-prompt", long_prompt,
        "--user-prompt", "Summarize this text",
        "--task", "Summarization",
        "--output", "/tmp/test_long_prompt.json",
        "--count", "1",
    ])
    # Should NOT crash with OSError; LLM client init may fail but that's ValueError
    assert "OSError" not in result.stdout
    assert result.exit_code != 0  # Will fail at LLM init, but not at resolution


def test_generate_tests_long_user_prompt_no_oserror():
    """Long inline user_prompt should NOT raise OSError in generate_tests."""
    long_prompt = "Summarize. " * 200  # ~3000 chars
    result = runner.invoke(app, [
        "generate-tests",
        "--system-prompt", "Be concise",
        "--user-prompt", long_prompt,
        "--task", "Summarization",
        "--output", "/tmp/test_long_user.json",
        "--count", "1",
    ])
    assert "OSError" not in result.stdout
    assert result.exit_code != 0


def test_report_winner_only_with_stats():
    """Verify winner-only mode still shows stats before winner info."""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()

    from abx.models import Experiment, Candidate, PromptPair
    from abx.storage import Storage

    storage = Storage(db_path=db.name)
    exp = Experiment(name="winner-stats", task_description="test")
    storage.save_experiment(exp)

    c = Candidate(
        prompts=PromptPair(system_prompt="Winning system", user_prompt="Winning user"),
        generation=5, scores=[9.5], composite_score=9.5,
        cost=0.002, latency=80.0, token_count=400,
    )
    storage.save_candidate(c, exp.id)
    storage.save_winner(exp.id, c.id, rank=1, generation=5)

    result = runner.invoke(app, [
        "report", "--experiment-id", exp.id, "--db", db.name, "--winner-only",
    ])

    assert result.exit_code == 0
    assert "Experiment Stats" in result.stdout
    assert "Winning Prompt" in result.stdout
    assert "Winning system" in result.stdout

    os.unlink(db.name)
