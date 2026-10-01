"""Tests for abx.self_optimizer.

Covers the meta-optimization module (GA over EVAL_SYSTEM_PROMPT variants).
Previously untested — this file closes the 13% coverage gap.
"""

import json
from unittest.mock import MagicMock

import pytest

from abx.models import (
    Candidate,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    PromptPair,
    TestCase,
    TestSuite,
)
from abx.self_optimizer import (
    SelfOptimizationRunner,
    compute_selfopt_composite_score,
    crossover_eval_prompts,
    evaluate_eval_candidate,
    evaluate_eval_prompt,
    evolve_eval_population,
    generate_initial_eval_population,
    mutate_eval_prompt,
    _parse_json_response,
    _strip_code_fence,
)
from abx.storage import Storage


def make_llm(initial_response=None, eval_score=7.0):
    """Mock LLM: returns initial population JSON then fixed eval scores."""
    client = MagicMock()
    call_count = [0]

    if initial_response is None:
        initial_response = json.dumps([
            {"system_prompt": "Variant A: be strict and analytical"},
            {"system_prompt": "Variant B: be lenient and holistic"},
            {"system_prompt": "Variant C: require chain-of-thought"},
        ])

    def side_effect(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            resp.content = initial_response
        else:
            resp.content = json.dumps({"score": eval_score, "reasoning": "ok"})
        resp.cost = 0.001
        resp.latency_ms = 100.0
        resp.total_tokens = 50
        return resp

    client.chat.side_effect = side_effect
    return client


def make_eval_llm(eval_score=7.0):
    """Mock LLM that ALWAYS returns a score JSON (for single-call helpers)."""
    client = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({"score": eval_score, "reasoning": "ok"})
    resp.cost = 0.001
    resp.latency_ms = 100.0
    resp.total_tokens = 50
    client.chat.return_value = resp
    return client


@pytest.fixture
def storage(tmp_path):
    s = Storage(db_path=str(tmp_path / "test.db"))
    yield s


def make_suite(expected=7.0):
    return TestSuite(
        task_description="Score AI responses against a rubric 0-10",
        test_cases=[
            TestCase(input="response one", rubric="be accurate", expected_score=expected),
            TestCase(input="response two", rubric="be complete", expected_score=expected),
        ],
    )


class TestStripCodeFence:
    def test_no_fence(self):
        assert _strip_code_fence("plain") == "plain"

    def test_json_fence(self):
        text = '```json\n{"system_prompt": "x"}\n```'
        assert _strip_code_fence(text) == '{"system_prompt": "x"}'

    def test_plain_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _strip_code_fence(text) == '{"a": 1}'


class TestParseJsonResponse:
    def test_direct_object(self):
        assert _parse_json_response('{"system_prompt": "x"}') == {"system_prompt": "x"}

    def test_direct_array(self):
        assert _parse_json_response('[{"a": 1}]') == [{"a": 1}]

    def test_fenced(self):
        text = '```json\n{"system_prompt": "y"}\n```'
        assert _parse_json_response(text) == {"system_prompt": "y"}

    def test_invalid_returns_none(self):
        assert _parse_json_response("garbage") is None


class TestGenerateInitialEvalPopulation:
    def test_generates_population(self):
        llm = make_llm()
        pop = generate_initial_eval_population(llm, "task", population_size=3)
        assert len(pop) == 3
        assert all(isinstance(c, Candidate) for c in pop)
        assert pop[0].prompts.system_prompt == "Variant A: be strict and analytical"
        assert pop[0].generation == 0

    def test_pads_with_baseline_when_short(self):
        llm = make_llm(initial_response=json.dumps([{"system_prompt": "Only one"}]))
        pop = generate_initial_eval_population(llm, "task", population_size=4)
        assert len(pop) == 4
        assert pop[1].prompts.system_prompt != ""

    def test_handles_single_object(self):
        llm = make_llm(initial_response=json.dumps({"system_prompt": "Single"}))
        pop = generate_initial_eval_population(llm, "task", population_size=1)
        assert len(pop) == 1
        assert pop[0].prompts.system_prompt == "Single"


class TestEvaluateEvalPrompt:
    def test_returns_accuracy_vs_expected(self):
        llm = make_eval_llm(eval_score=7.0)
        tc = TestCase(input="some response", rubric="rubric", expected_score=10.0)
        result = evaluate_eval_prompt(llm, "eval sys prompt", tc, "task")
        assert result["score"] == 7.0
        assert result["expected"] == 10.0
        assert result["accuracy"] == pytest.approx(0.7, abs=0.01)
        assert result["cost"] > 0
        assert result["latency"] > 0
        assert result["token_count"] > 0

    def test_defaults_to_mid_score_on_bad_json(self):
        llm = MagicMock()
        resp = MagicMock()
        resp.content = "I cannot produce JSON"
        resp.cost = 0.001
        resp.latency_ms = 50.0
        resp.total_tokens = 10
        llm.chat.return_value = resp

        tc = TestCase(input="x", rubric="y", expected_score=10.0)
        result = evaluate_eval_prompt(llm, "sys", tc, "task")
        assert result["score"] == 5.0
        assert result["accuracy"] == pytest.approx(0.5, abs=0.01)

    def test_parses_fenced_json(self):
        llm = make_eval_llm(eval_score=7.0)
        llm.chat.return_value.content = '```json\n{"score": 3, "reasoning": "r"}\n```'
        tc = TestCase(input="x", rubric="y", expected_score=3.0)
        result = evaluate_eval_prompt(llm, "sys", tc, "task")
        assert result["score"] == 3.0
        assert result["accuracy"] == pytest.approx(1.0)

    def test_expected_score_zero_is_not_treated_as_missing(self):
        llm = make_eval_llm(eval_score=0.0)
        tc = TestCase(input="x", rubric="y", expected_score=0.0)
        result = evaluate_eval_prompt(llm, "sys", tc, "task")
        assert result["expected"] == 0.0
        assert result["accuracy"] == pytest.approx(1.0)


class TestEvaluateEvalCandidate:
    def test_scores_all_test_cases(self):
        llm = make_eval_llm(eval_score=8.0)
        cand = Candidate(prompts=PromptPair(system_prompt="sys", user_prompt=""))
        cand, mean_acc = evaluate_eval_candidate(llm, cand, make_suite(expected=10.0))
        assert len(cand.scores) == 2
        assert cand.scores[0] == pytest.approx(0.8, abs=0.01)
        assert mean_acc == pytest.approx(0.8, abs=0.01)
        assert cand.cost > 0
        assert cand.latency > 0


class TestComputeSelfoptCompositeScore:
    def test_high_accuracy_scores_well(self):
        cand = Candidate(
            prompts=PromptPair(system_prompt="s"),
            scores=[0.9, 0.8],
            cost=0.001,
            latency=100.0,
        )
        config = ExperimentConfig(kpi_weights={"accuracy": 0.6, "cost": 0.2, "latency": 0.2})
        score = compute_selfopt_composite_score(cand, config, baseline_cost=0.002, baseline_latency=200.0)
        assert score == pytest.approx(0.71, abs=0.01)

    def test_zero_accuracy(self):
        cand = Candidate(prompts=PromptPair(system_prompt="s"), scores=[0.0])
        config = ExperimentConfig(kpi_weights={"accuracy": 0.6, "cost": 0.2, "latency": 0.2})
        score = compute_selfopt_composite_score(cand, config)
        assert score < 0.6


class TestMutateEvalPrompt:
    def test_mutates_when_rate_one(self):
        cand = Candidate(prompts=PromptPair(system_prompt="original", user_prompt=""))
        llm = MagicMock()
        resp = MagicMock()
        resp.content = json.dumps({"system_prompt": "mutated"})
        llm.chat.return_value = resp

        mutated = mutate_eval_prompt(llm, cand, "task", mutation_rate=1.0)
        assert mutated.prompts.system_prompt == "mutated"
        assert mutated.parent_id == cand.id
        assert mutated.mutation_type == "llm_mutation"

    def test_no_mutation_when_rate_zero(self):
        cand = Candidate(prompts=PromptPair(system_prompt="original", user_prompt=""))
        mutated = mutate_eval_prompt(MagicMock(), cand, "task", mutation_rate=0.0)
        assert mutated.prompts.system_prompt == "original"
        assert mutated.mutation_type == "no_mutation"


class TestCrossoverEvalPrompts:
    def test_llm_crossover(self):
        a = Candidate(prompts=PromptPair(system_prompt="A", user_prompt=""))
        b = Candidate(prompts=PromptPair(system_prompt="B", user_prompt=""))
        llm = MagicMock()
        resp = MagicMock()
        resp.content = json.dumps({"system_prompt": "AB combined"})
        llm.chat.return_value = resp

        child = crossover_eval_prompts(a, b, "task", llm_client=llm)
        assert child.prompts.system_prompt == "AB combined"
        assert child.mutation_type == "crossover_llm"

    def test_swap_fallback(self):
        a = Candidate(prompts=PromptPair(system_prompt="A", user_prompt=""))
        b = Candidate(prompts=PromptPair(system_prompt="B", user_prompt=""))
        child = crossover_eval_prompts(a, b, "task", llm_client=None)
        assert child.prompts.system_prompt == "A"
        assert child.mutation_type == "crossover_swap"


class TestEvolveEvalPopulation:
    def test_evolves_generation(self):
        pop = [
            Candidate(prompts=PromptPair(system_prompt=f"S{i}", user_prompt=""), composite_score=float(i), generation=0)
            for i in range(1, 6)
        ]
        llm = make_llm()
        config = ExperimentConfig(population_size=5, crossover_rate=0.0, mutation_rate=1.0)
        nxt = evolve_eval_population(llm, pop, "task", config)
        assert len(nxt) == config.population_size
        assert all(c.generation == 1 for c in nxt)
        assert nxt[0].mutation_type == "elite"
        assert nxt[0].prompts.system_prompt == "S5"


class TestSelfOptimizationRunner:
    def test_run_completes(self, storage):
        exp = Experiment(
            name="self-opt-test",
            task_description="Score responses 0-10",
            test_suite=make_suite(),
            config=ExperimentConfig(cycles=2, population_size=3),
        )
        storage.save_experiment(exp)
        llm = make_llm()
        runner = SelfOptimizationRunner(exp, storage, llm)
        result = runner.run()
        assert result.status in (ExperimentStatus.COMPLETED, ExperimentStatus.CONVERGED)
        assert result.current_generation >= 1
        assert len(result.winners) >= 1

    def test_run_saves_candidates(self, storage):
        exp = Experiment(
            name="self-opt-save",
            task_description="Score responses",
            test_suite=make_suite(),
            config=ExperimentConfig(cycles=1, population_size=2),
        )
        storage.save_experiment(exp)
        llm = make_llm()
        runner = SelfOptimizationRunner(exp, storage, llm)
        runner.run()
        candidates = storage.get_candidates(exp.id)
        assert len(candidates) >= 2

    def test_convergence_sets_current_generation(self, storage):
        """BUG-REGRESSION: when the run converges at generation G, the
        experiment's current_generation must equal G (not G-1)."""
        exp = Experiment(
            name="self-opt-conv",
            task_description="Score responses",
            test_suite=make_suite(),
            config=ExperimentConfig(
                cycles=5,
                population_size=2,
                plateau_threshold=0.02,
                plateau_rounds=1,
            ),
        )
        storage.save_experiment(exp)
        llm = make_llm(eval_score=7.0)
        runner = SelfOptimizationRunner(exp, storage, llm)
        result = runner.run()
        assert result.status == ExperimentStatus.CONVERGED
        assert result.current_generation >= 2

    def test_evaluate_baseline(self, storage):
        exp = Experiment(
            name="self-opt-base",
            task_description="Score responses",
            test_suite=make_suite(),
        )
        storage.save_experiment(exp)
        llm = make_llm(eval_score=7.0)
        runner = SelfOptimizationRunner(exp, storage, llm)
        result = runner.evaluate_baseline()
        assert "accuracy" in result
        assert "cost" in result
        assert "latency" in result
        assert len(result["scores"]) == 2


class TestExperimentRunnerConvergenceGeneration:
    def test_convergence_sets_current_generation(self, storage):
        """BUG-REGRESSION: ExperimentRunner must also record the generation
        at which convergence was detected."""
        from abx.experiment import ExperimentRunner

        suite = make_suite()
        exp = Experiment(
            name="exp-conv",
            task_description="Score responses",
            test_suite=suite,
            config=ExperimentConfig(
                cycles=5,
                population_size=2,
                plateau_threshold=0.02,
                plateau_rounds=1,
            ),
        )
        storage.save_experiment(exp)

        llm = MagicMock()
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
                resp.content = json.dumps({"score": 7, "reasoning": "ok"})
            resp.cost = 0.001
            resp.latency_ms = 100.0
            resp.total_tokens = 50
            return resp

        llm.chat.side_effect = side_effect

        runner = ExperimentRunner(exp, storage, llm)
        result = runner.run()
        assert result.status == ExperimentStatus.CONVERGED
        assert result.current_generation >= 2
