"""Tests for abx.experiment."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from abx.experiment import ExperimentRunner
from abx.llm import DeepSeekClient
from abx.models import (
    Candidate,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    PromptPair,
    TestCase,
    TestSuite,
)
from abx.storage import Storage


@pytest.fixture
def storage():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = Storage(db_path=tmp.name)
    yield s
    os.unlink(tmp.name)


def make_mock_llm():
    """Create a mock LLM that returns canned responses."""
    client = MagicMock()
    call_count = [0]
    responses = [
        # Initial population generation
        json.dumps([
            {"system_prompt": "Be concise", "user_prompt": "Summarize the input"},
            {"system_prompt": "Be thorough", "user_prompt": "Explain in detail"},
            {"system_prompt": "Be creative", "user_prompt": "Think outside the box"},
            {"system_prompt": "Be factual", "user_prompt": "Provide accurate info"},
            {"system_prompt": "Be helpful", "user_prompt": "Assist the user"},
        ]),
    ]
    # Add a generic response for all LLM calls
    for _ in range(200):
        responses.append(json.dumps({"score": 7, "reasoning": "Good response"}))

    def side_effect(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(responses):
            resp_text = responses[idx]
        else:
            resp_text = json.dumps({"score": 7})
        r = MagicMock()
        r.content = resp_text
        r.cost = 0.001
        r.latency_ms = 100.0
        r.total_tokens = 50
        return r

    client.chat.side_effect = side_effect
    return client


class TestExperimentRunner:
    def test_run_completes(self, storage):
        suite = TestSuite(
            task_description="Summarize text",
            test_cases=[
                TestCase(input="Long article text", rubric="Must be concise"),
                TestCase(input="Another text", rubric="Must capture key points"),
            ],
        )
        exp = Experiment(
            name="test-run",
            task_description="Summarize text",
            test_suite=suite,
            config=ExperimentConfig(cycles=2, population_size=3),
        )
        storage.save_experiment(exp)
        llm = make_mock_llm()
        runner = ExperimentRunner(exp, storage, llm)
        result = runner.run()
        assert result.status in (ExperimentStatus.COMPLETED, ExperimentStatus.CONVERGED)
        assert result.current_generation >= 1

    def test_run_saves_candidates(self, storage):
        suite = TestSuite(
            task_description="Extract data",
            test_cases=[TestCase(input="data here", rubric="Be accurate")],
        )
        exp = Experiment(
            name="test-save",
            task_description="Extract data",
            test_suite=suite,
            config=ExperimentConfig(cycles=1, population_size=2),
        )
        storage.save_experiment(exp)
        llm = make_mock_llm()
        runner = ExperimentRunner(exp, storage, llm)
        runner.run()
        candidates = storage.get_candidates(exp.id)
        assert len(candidates) >= 2

    def test_convergence_detection(self):
        runner = object.__new__(ExperimentRunner)
        runner._convergence_history = [0.9, 0.91, 0.905, 0.908, 0.903, 0.906]
        config = ExperimentConfig(plateau_threshold=0.02, plateau_rounds=5)
        # Last 5 values: 0.91, 0.905, 0.908, 0.903, 0.906 — range = 0.007 < 0.02
        assert runner._check_convergence(config) is True

    def test_no_convergence_when_not_plateaued(self):
        runner = object.__new__(ExperimentRunner)
        runner._convergence_history = [0.5, 0.6, 0.7, 0.8, 0.81, 0.82]
        config = ExperimentConfig(plateau_threshold=0.02, plateau_rounds=5)
        # Last 5: 0.6, 0.7, 0.8, 0.81, 0.82 — range = 0.22 > 0.02
        assert runner._check_convergence(config) is False

    def test_not_enough_history_for_convergence(self):
        runner = object.__new__(ExperimentRunner)
        runner._convergence_history = [0.9, 0.91]
        config = ExperimentConfig(plateau_threshold=0.02, plateau_rounds=5)
        assert runner._check_convergence(config) is False
