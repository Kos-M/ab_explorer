"""End-to-end CLI verification with a mocked LLM client.

Covers the full lifecycle: init -> run -> report -> list-experiments
against a real SQLite DB, without making live API calls.
"""

import json
import os
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from abx.cli import app

runner = CliRunner()


def make_fake_claude():
    """Fake LLM client: init population JSON, then fixed eval scores."""
    fake = MagicMock()
    fake.model = "haiku"
    call_count = [0]
    init_resp = json.dumps([
        {"system_prompt": "S1", "user_prompt": "U1"},
        {"system_prompt": "S2", "user_prompt": "U2"},
    ])

    def side_effect(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            resp.content = init_resp
        else:
            resp.content = json.dumps({"score": 8, "reasoning": "good"})
        resp.cost = 0.001
        resp.latency_ms = 100.0
        resp.total_tokens = 50
        return resp

    fake.chat.side_effect = side_effect
    return fake


def test_cli_full_lifecycle(tmp_path):
    tests = tmp_path / "tests.json"
    tests.write_text(json.dumps({
        "task_description": "Extract dates from text",
        "evaluation_model": "haiku",
        "test_cases": [
            {"input": "Event on March 5, 2024", "rubric": "Must extract exact date"},
            {"input": "Deadline 2024-12-31", "rubric": "Must extract ISO date"},
        ],
    }))
    db = tmp_path / "e2e.db"

    # 1. init
    r = runner.invoke(app, [
        "init",
        "--task", "Extract dates from text",
        "--tests", str(tests),
        "--output", str(db),
        "--name", "e2e-test",
    ])
    assert r.exit_code == 0, r.stdout
    assert "Experiment initialized" in r.stdout
    import re
    match = re.search(r"Experiment initialized: ([0-9a-f]{12})", r.stdout)
    assert match, r.stdout
    exp_id = match.group(1)

    # 2. run with mocked LLM
    with patch("abx.cli.make_client", return_value=make_fake_claude()):
        r = runner.invoke(app, [
            "run",
            "--experiment-id", exp_id,
            "--db", str(db),
            "--cycles", "2",
            "--population", "2",
        ])
    assert r.exit_code == 0, r.stdout
    assert "Experiment complete" in r.stdout

    # 3. report
    r = runner.invoke(app, [
        "report",
        "--experiment-id", exp_id,
        "--db", str(db),
    ])
    assert r.exit_code == 0, r.stdout
    assert "Experiment Stats" in r.stdout
    assert "Total Cost" in r.stdout
    assert "Total Tokens" in r.stdout
    assert "LLM Calls" in r.stdout

    # 4. winner-only report
    r = runner.invoke(app, [
        "report",
        "--experiment-id", exp_id,
        "--db", str(db),
        "--winner-only",
    ])
    assert r.exit_code == 0, r.stdout
    assert "Winning Prompt" in r.stdout

    # 5. list-experiments
    r = runner.invoke(app, [
        "list-experiments",
        "--db", str(db),
    ])
    assert r.exit_code == 0, r.stdout
    assert "e2e-test" in r.stdout

    # 6. persisted state is consistent
    from abx.storage import Storage
    storage = Storage(db_path=str(db))
    exp = storage.get_experiment(exp_id)
    assert exp is not None
    assert exp.status.value in ("completed", "converged")
    candidates = storage.get_candidates(exp_id)
    assert len(candidates) >= 2
    winners = storage.get_winners(exp_id)
    assert len(winners) >= 1


def test_cli_run_missing_experiment(tmp_path):
    db = tmp_path / "missing.db"
    r = runner.invoke(app, [
        "run",
        "--experiment-id", "does-not-exist",
        "--db", str(db),
    ])
    assert r.exit_code != 0
    assert "not found" in r.stdout


def _write_selfopt_tests(tmp_path):
    tests = tmp_path / "selfopt_tests.json"
    tests.write_text(json.dumps({
        "task_description": "Score AI responses against a rubric 0-10",
        "evaluation_model": "haiku",
        "test_cases": [
            {"input": "The Eiffel Tower is in Paris.", "rubric": "Factual accuracy", "expected_score": 9.0},
            {"input": "Capital of Australia is Sydney.", "rubric": "Factual accuracy", "expected_score": 3.0},
        ],
    }))
    return tests


def test_cli_self_optimize_lifecycle(tmp_path):
    tests = _write_selfopt_tests(tmp_path)
    db = tmp_path / "selfopt.db"

    with patch("abx.cli.make_client", return_value=make_fake_claude()):
        r = runner.invoke(app, [
            "self-optimize",
            "--tests", str(tests),
            "--task", "Score responses 0-10",
            "--name", "selfopt-e2e",
            "--db", str(db),
            "--cycles", "2",
            "--population", "2",
        ])
    assert r.exit_code == 0, r.stdout
    assert "Self-optimization complete" in r.stdout
    assert "selfopt-e2e" in r.stdout or "Self-optimization experiment initialized" in r.stdout

    # DB has experiment + candidates
    from abx.storage import Storage
    storage = Storage(db_path=str(db))
    experiments = storage.list_experiments()
    assert len(experiments) == 1
    exp = storage.get_experiment(experiments[0]["id"])
    assert exp.status.value in ("completed", "converged")
    assert len(storage.get_candidates(exp.id)) >= 2


def test_cli_self_optimize_missing_file(tmp_path):
    db = tmp_path / "nope.db"
    r = runner.invoke(app, [
        "self-optimize",
        "--tests", str(tmp_path / "missing.json"),
        "--db", str(db),
    ])
    assert r.exit_code != 0
    assert "not found" in r.stdout


def test_cli_self_eval_compare(tmp_path):
    # First: build an experiment with winners via self-optimize
    tests = _write_selfopt_tests(tmp_path)
    db = tmp_path / "cmp.db"
    with patch("abx.cli.make_client", return_value=make_fake_claude()):
        r = runner.invoke(app, [
            "self-optimize",
            "--tests", str(tests),
            "--name", "cmp-exp",
            "--db", str(db),
            "--cycles", "1",
            "--population", "2",
        ])
    assert r.exit_code == 0, r.stdout

    from abx.storage import Storage
    storage = Storage(db_path=str(db))
    exp_id = storage.list_experiments()[0]["id"]

    # Now compare baseline vs optimized
    with patch("abx.cli.make_client", return_value=make_fake_claude()):
        r = runner.invoke(app, [
            "self-eval-compare",
            "--experiment-id", exp_id,
            "--db", str(db),
        ])
    assert r.exit_code == 0, r.stdout
    assert "Baseline EVAL_SYSTEM_PROMPT" in r.stdout
    assert "Winning EVAL_SYSTEM_PROMPT" in r.stdout
    assert "Comparison" in r.stdout or "Accuracy" in r.stdout


def test_cli_generate_tests_mocked(tmp_path):
    """generate-tests writes a valid tests.json via mocked LLM."""
    out = tmp_path / "gen_tests.json"
    fake = MagicMock()
    fake.model = "haiku"
    resp = MagicMock()
    resp.content = json.dumps({
        "test_cases": [
            {"input": "sample input 1", "rubric": "must be accurate"},
            {"input": "sample input 2", "rubric": "must be concise"},
        ]
    })
    fake.chat.return_value = resp

    with patch("abx.cli.make_client", return_value=fake):
        r = runner.invoke(app, [
            "generate-tests",
            "--system-prompt", "You are a helpful assistant",
            "--user-prompt", "Answer the question",
            "--task", "QA",
            "--output", str(out),
            "--count", "2",
        ])
    assert r.exit_code == 0, r.stdout
    assert "Generated 2 test cases" in r.stdout
    data = json.loads(out.read_text())
    assert len(data["test_cases"]) == 2
    assert data["task_description"] == "QA"
