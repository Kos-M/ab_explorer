"""Composite KPI scoring for ab_explorer."""

from __future__ import annotations

from typing import Optional

from .models import Candidate, ExperimentConfig


def compute_composite_score(
    candidate: Candidate,
    config: ExperimentConfig,
    baseline_cost: Optional[float] = None,
    baseline_latency: Optional[float] = None,
) -> float:
    """Compute the composite KPI score for a candidate.

    Formula:
        composite = w_a * normalized_accuracy
                  + w_c * (1 - relative_cost)
                  + w_l * (1 - relative_latency)

    Where:
        normalized_accuracy = avg_score / 10.0  (map 0-10 to 0-1)
        relative_cost = cost / max(baseline_cost, candidate.cost)
        relative_latency = latency / max(baseline_latency, candidate.latency)

    Weights (w_a, w_c, w_l) come from config.kpi_weights.
    """
    weights = config.kpi_weights
    w_a = weights.get("accuracy", 0.5)
    w_c = weights.get("cost", 0.3)
    w_l = weights.get("latency", 0.2)

    # Normalized accuracy (0-10 → 0-1)
    accuracy = candidate.avg_score() / 10.0

    # Relative cost (lower is better)
    if baseline_cost and baseline_cost > 0:
        rel_cost = candidate.cost / max(baseline_cost, candidate.cost)
    else:
        rel_cost = candidate.cost / max(candidate.cost, 0.001)
    cost_factor = 1.0 - min(rel_cost, 1.0)

    # Relative latency (lower is better)
    if baseline_latency and baseline_latency > 0:
        rel_latency = candidate.latency / max(baseline_latency, candidate.latency)
    else:
        rel_latency = candidate.latency / max(candidate.latency, 0.001)
    latency_factor = 1.0 - min(rel_latency, 1.0)

    score = (w_a * accuracy) + (w_c * cost_factor) + (w_l * latency_factor)
    return round(score, 4)


def score_population(
    population: list[Candidate],
    config: ExperimentConfig,
) -> list[Candidate]:
    """Score all candidates in a population using composite KPI.

    Uses the population's best performer as baseline for relative cost/latency.
    """
    if not population:
        return population

    # Find baseline (max cost/latency among population)
    baseline_cost = max(c.cost for c in population) if population else None
    baseline_latency = max(c.latency for c in population) if population else None

    for candidate in population:
        candidate.composite_score = compute_composite_score(
            candidate, config, baseline_cost, baseline_latency
        )

    return population


def find_best_candidate(population: list[Candidate]) -> Optional[Candidate]:
    """Return the candidate with the highest composite score."""
    if not population:
        return None
    return max(population, key=lambda c: c.composite_score)
