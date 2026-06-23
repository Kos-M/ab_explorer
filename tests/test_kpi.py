"""Tests for abx.kpi."""

import pytest

from abx.kpi import compute_composite_score, find_best_candidate, score_population
from abx.models import Candidate, ExperimentConfig, PromptPair


class TestComputeCompositeScore:
    def test_perfect_candidate(self):
        cand = Candidate(
            prompts=PromptPair(),
            scores=[10.0, 10.0],
            cost=0.0,
            latency=0.0,
        )
        config = ExperimentConfig()
        score = compute_composite_score(cand, config)
        # Perfect accuracy (1.0) * 0.5 + cost factor (1.0) * 0.3 + latency factor (1.0) * 0.2
        assert score == pytest.approx(0.5 + 0.3 + 0.2, rel=0.01)
        assert score <= 1.0

    def test_worst_candidate(self):
        cand = Candidate(
            prompts=PromptPair(),
            scores=[0.0],
            cost=1.0,
            latency=1.0,
        )
        config = ExperimentConfig()
        score = compute_composite_score(cand, config)
        # accuracy=0, cost_factor=0 (self is baseline), latency_factor=0
        assert score == 0.0

    def test_mid_candidate(self):
        cand = Candidate(
            prompts=PromptPair(),
            scores=[5.0],
            cost=0.5,
            latency=0.5,
        )
        config = ExperimentConfig()
        score = compute_composite_score(cand, config)
        # accuracy=0.5 -> 0.5*0.5 = 0.25
        # cost_factor = 1 - 0.5/0.5 = 0 -> 0*0.3 = 0
        # latency_factor = 1 - 0.5/0.5 = 0 -> 0*0.2 = 0
        # total = 0.25
        assert score == pytest.approx(0.25, rel=0.01)

    def test_custom_weights(self):
        cand = Candidate(
            prompts=PromptPair(),
            scores=[10.0],
            cost=0.0,
            latency=0.0,
        )
        config = ExperimentConfig(kpi_weights={"accuracy": 1.0, "cost": 0.0, "latency": 0.0})
        score = compute_composite_score(cand, config)
        assert score == pytest.approx(1.0, rel=0.01)

    def test_with_baseline(self):
        cand = Candidate(
            prompts=PromptPair(),
            scores=[8.0],
            cost=0.002,
            latency=0.3,
        )
        config = ExperimentConfig()
        score = compute_composite_score(cand, config, baseline_cost=0.004, baseline_latency=0.6)
        # accuracy = 8/10 = 0.8 -> 0.8*0.5 = 0.4
        # cost_factor = 1 - 0.002/0.004 = 0.5 -> 0.5*0.3 = 0.15
        # latency_factor = 1 - 0.3/0.6 = 0.5 -> 0.5*0.2 = 0.1
        # total = 0.65
        assert score == pytest.approx(0.65, rel=0.01)


class TestScorePopulation:
    def test_scores_all_candidates(self):
        pop = [
            Candidate(prompts=PromptPair(), scores=[9.0], cost=0.001, latency=0.1),
            Candidate(prompts=PromptPair(), scores=[7.0], cost=0.003, latency=0.4),
            Candidate(prompts=PromptPair(), scores=[5.0], cost=0.005, latency=0.8),
        ]
        config = ExperimentConfig()
        scored = score_population(pop, config)
        assert all(c.composite_score > 0 for c in scored)
        # Best candidate should have highest score
        assert scored[0].composite_score > scored[2].composite_score


class TestFindBestCandidate:
    def test_finds_best(self):
        pop = [
            Candidate(prompts=PromptPair(), composite_score=5.0),
            Candidate(prompts=PromptPair(), composite_score=9.0),
            Candidate(prompts=PromptPair(), composite_score=7.0),
        ]
        best = find_best_candidate(pop)
        assert best.composite_score == 9.0

    def test_empty_returns_none(self):
        assert find_best_candidate([]) is None
