"""Tests for abx.models."""

import json
from abx.models import (
    Candidate,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    PromptPair,
    PromptType,
    TestCase,
    TestSuite,
)


class TestTestCase:
    def test_basic(self):
        tc = TestCase(input="hello", rubric="must respond politely")
        assert tc.input == "hello"
        assert tc.rubric == "must respond politely"

    def test_serialization(self):
        tc = TestCase(input="test", rubric="must work")
        data = tc.model_dump()
        assert data["input"] == "test"
        restored = TestCase(**data)
        assert restored == tc


class TestTestSuite:
    def test_basic(self):
        suite = TestSuite(
            task_description="extract dates",
            test_cases=[TestCase(input="March 5", rubric="return date")],
        )
        assert suite.task_description == "extract dates"
        assert len(suite.test_cases) == 1

    def test_default_model(self):
        suite = TestSuite(task_description="x", test_cases=[])
        assert suite.evaluation_model == "deepseek-chat"


class TestPromptPair:
    def test_defaults(self):
        p = PromptPair()
        assert p.system_prompt == ""
        assert p.user_prompt == ""

    def test_full(self):
        p = PromptPair(system_prompt="Be helpful", user_prompt="Hello")
        assert p.system_prompt == "Be helpful"
        assert p.user_prompt == "Hello"


class TestCandidate:
    def test_defaults(self):
        c = Candidate(prompts=PromptPair())
        assert c.id is not None
        assert len(c.id) == 12
        assert c.generation == 0
        assert c.scores == []
        assert c.composite_score == 0.0

    def test_avg_score_empty(self):
        c = Candidate(prompts=PromptPair())
        assert c.avg_score() == 0.0

    def test_avg_score(self):
        c = Candidate(prompts=PromptPair(), scores=[8.0, 9.0, 7.0])
        assert c.avg_score() == 8.0

    def test_serialization_roundtrip(self):
        c = Candidate(
            prompts=PromptPair(system_prompt="sys", user_prompt="user"),
            generation=1,
            scores=[8.5, 9.0],
            composite_score=8.7,
            cost=0.001,
            latency=0.5,
            token_count=150,
        )
        data = c.model_dump()
        restored = Candidate(**data)
        assert restored.id == c.id
        assert restored.prompts.system_prompt == "sys"
        assert restored.scores == [8.5, 9.0]


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.cycles == 20
        assert cfg.population_size == 5
        assert cfg.kpi_weights["accuracy"] == 0.5
        assert cfg.kpi_weights["cost"] == 0.3
        assert cfg.kpi_weights["latency"] == 0.2

    def test_mutation_rates(self):
        cfg = ExperimentConfig(mutation_rate=0.5, crossover_rate=0.7)
        assert cfg.mutation_rate == 0.5
        assert cfg.crossover_rate == 0.7

    def test_serialization(self):
        cfg = ExperimentConfig(cycles=10)
        json_str = cfg.model_dump_json()
        data = json.loads(json_str)
        assert data["cycles"] == 10
        restored = ExperimentConfig(**data)
        assert restored.cycles == 10


class TestExperiment:
    def test_defaults(self):
        exp = Experiment(name="test-exp", task_description="test task")
        assert exp.id is not None
        assert exp.status == ExperimentStatus.CREATED
        assert exp.current_generation == 0
        assert exp.winners == []

    def test_status_transitions(self):
        exp = Experiment(name="x", task_description="y")
        exp.status = ExperimentStatus.RUNNING
        assert exp.status == ExperimentStatus.RUNNING
        exp.status = ExperimentStatus.COMPLETED
        assert exp.status == ExperimentStatus.COMPLETED

    def test_with_config(self):
        cfg = ExperimentConfig(cycles=5)
        exp = Experiment(name="x", task_description="y", config=cfg)
        assert exp.config.cycles == 5
