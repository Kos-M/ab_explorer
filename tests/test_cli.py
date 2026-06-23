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
        "evaluation_model": "deepseek-chat",
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
        "evaluation_model": "deepseek-chat",
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
