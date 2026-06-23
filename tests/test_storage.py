"""Tests for abx.storage."""

import os
import tempfile

import pytest

from abx.models import Candidate, Experiment, ExperimentConfig, PromptPair, TestCase, TestSuite
from abx.storage import Storage


@pytest.fixture
def storage():
    """Create a Storage instance backed by a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = Storage(db_path=tmp.name)
    yield s
    os.unlink(tmp.name)


class TestStorageExperiment:
    def test_save_and_load(self, storage):
        exp = Experiment(name="test-exp", task_description="extract dates")
        storage.save_experiment(exp)
        loaded = storage.get_experiment(exp.id)
        assert loaded is not None
        assert loaded.id == exp.id
        assert loaded.name == "test-exp"
        assert loaded.task_description == "extract dates"
        assert loaded.status.value == "created"

    def test_save_with_test_suite(self, storage):
        suite = TestSuite(
            task_description="extract dates",
            test_cases=[TestCase(input="March 5", rubric="return date")],
        )
        exp = Experiment(name="suite-test", task_description="x", test_suite=suite)
        storage.save_experiment(exp)
        loaded = storage.get_experiment(exp.id)
        assert loaded.test_suite is not None
        assert loaded.test_suite.task_description == "extract dates"
        assert len(loaded.test_suite.test_cases) == 1

    def test_save_with_config(self, storage):
        cfg = ExperimentConfig(cycles=5, population_size=3)
        exp = Experiment(name="cfg-test", task_description="x", config=cfg)
        storage.save_experiment(exp)
        loaded = storage.get_experiment(exp.id)
        assert loaded.config.cycles == 5
        assert loaded.config.population_size == 3

    def test_list_experiments(self, storage):
        exp1 = Experiment(name="exp1", task_description="a")
        exp2 = Experiment(name="exp2", task_description="b")
        storage.save_experiment(exp1)
        storage.save_experiment(exp2)
        items = storage.list_experiments()
        assert len(items) >= 2

    def test_delete_experiment(self, storage):
        exp = Experiment(name="delete-me", task_description="x")
        storage.save_experiment(exp)
        storage.delete_experiment(exp.id)
        assert storage.get_experiment(exp.id) is None


class TestStorageCandidate:
    def test_save_and_load(self, storage):
        exp = Experiment(name="cand-test", task_description="x")
        storage.save_experiment(exp)

        cand = Candidate(
            prompts=PromptPair(system_prompt="Be concise", user_prompt="Hello"),
            generation=0,
            scores=[8.0, 7.5],
            composite_score=7.8,
        )
        storage.save_candidate(cand, exp.id)

        candidates = storage.get_candidates(exp.id)
        assert len(candidates) == 1
        assert candidates[0].id == cand.id
        assert candidates[0].prompts.system_prompt == "Be concise"
        assert candidates[0].scores == [8.0, 7.5]

    def test_filter_by_generation(self, storage):
        exp = Experiment(name="gen-test", task_description="x")
        storage.save_experiment(exp)

        c1 = Candidate(prompts=PromptPair(), generation=0, scores=[5.0])
        c2 = Candidate(prompts=PromptPair(), generation=1, scores=[9.0])
        c3 = Candidate(prompts=PromptPair(), generation=1, scores=[8.0])
        storage.save_candidate(c1, exp.id)
        storage.save_candidate(c2, exp.id)
        storage.save_candidate(c3, exp.id)

        gen0 = storage.get_candidates(exp.id, generation=0)
        assert len(gen0) == 1

        gen1 = storage.get_candidates(exp.id, generation=1)
        assert len(gen1) == 2


class TestStorageWinners:
    def test_save_and_get_winners(self, storage):
        exp = Experiment(name="winner-test", task_description="x")
        storage.save_experiment(exp)

        c1 = Candidate(prompts=PromptPair(), generation=4, composite_score=9.5)
        c2 = Candidate(prompts=PromptPair(), generation=4, composite_score=9.0)
        storage.save_candidate(c1, exp.id)
        storage.save_candidate(c2, exp.id)

        storage.save_winner(exp.id, c1.id, rank=1, generation=4)
        storage.save_winner(exp.id, c2.id, rank=2, generation=4)

        winners = storage.get_winners(exp.id)
        assert len(winners) == 2


class TestStorageTestResult:
    def test_save_result(self, storage):
        exp = Experiment(name="result-test", task_description="x")
        storage.save_experiment(exp)
        cand = Candidate(prompts=PromptPair())
        storage.save_candidate(cand, exp.id)

        storage.save_test_result(
            candidate_id=cand.id,
            test_case_index=0,
            score=8.5,
            output="extracted date: March 5",
            cost=0.0005,
            latency=0.3,
            token_count=100,
        )
        # No crash = success. Read-back is via candidates.
        assert True
