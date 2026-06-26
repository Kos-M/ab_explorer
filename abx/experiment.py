"""Core optimization loop for ab_explorer."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .evaluator import evaluate_candidate
from .kpi import find_best_candidate, score_population
from .llm import DeepSeekClient
from .models import (
    Candidate,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    PromptPair,
    TestSuite,
)
from .population import evolve_population, generate_initial_population
from .storage import Storage

console = Console()


class ExperimentRunner:
    """Runs the prompt optimization experiment loop."""

    def __init__(
        self,
        experiment: Experiment,
        storage: Storage,
        llm_client: DeepSeekClient,
    ):
        self.experiment = experiment
        self.storage = storage
        self.llm = llm_client
        self._convergence_history: list[float] = []

    def run(self) -> Experiment:
        """Execute the full optimization loop."""
        self.experiment.status = ExperimentStatus.RUNNING
        self.experiment.updated_at = datetime.now()
        self.storage.save_experiment(self.experiment)

        config = self.experiment.config
        test_suite = self.experiment.test_suite
        if not test_suite:
            raise ValueError("Experiment has no test suite configured")

        console.print(f"\n[bold cyan]🚀 Starting experiment:[/] {self.experiment.name}")
        console.print(f"[dim]Cycles: {config.cycles} | Population: {config.population_size}[/]\n")

        # Generation 0: initial population
        console.print("[bold]Generation 0:[/] Generating initial population...")
        population = generate_initial_population(
            self.llm,
            self.experiment.task_description,
            config.population_size,
            seed_system_prompt=config.system_prompt,
        )
        self._evaluate_and_score(population, test_suite, config)
        self._save_generation(population, 0)
        self._print_gen_summary(0, population)

        # Evolve through generations
        for gen in range(1, config.cycles + 1):
            console.print(f"[bold]Generation {gen}:[/] Evolving...")
            population = evolve_population(
                self.llm, population, self.experiment.task_description, config
            )
            self._evaluate_and_score(population, test_suite, config)
            self._save_generation(population, gen)
            self._print_gen_summary(gen, population)

            # Track best score for convergence
            best = find_best_candidate(population)
            if best:
                self._convergence_history.append(best.composite_score)
                self.experiment.winners.append(best)
                self.storage.save_winner(
                    self.experiment.id, best.id, rank=1, generation=gen
                )

            # Check convergence
            if self._check_convergence(config):
                console.print(f"[bold green]✓ Converged at generation {gen}![/]")
                self.experiment.status = ExperimentStatus.CONVERGED
                break

            self.experiment.current_generation = gen
            self.experiment.updated_at = datetime.now()
            self.storage.save_experiment(self.experiment)

        # Finalize
        if self.experiment.status != ExperimentStatus.CONVERGED:
            self.experiment.status = ExperimentStatus.COMPLETED
        self.experiment.current_generation = (
            min(config.cycles, self.experiment.current_generation)
        )
        self.experiment.updated_at = datetime.now()
        self.storage.save_experiment(self.experiment)

        self._print_final_summary()
        return self.experiment

    def _evaluate_and_score(
        self,
        population: list[Candidate],
        test_suite: TestSuite,
        config: ExperimentConfig,
    ) -> None:
        """Evaluate all candidates and compute composite scores."""
        for i, candidate in enumerate(population):
            console.print(f"  Evaluating candidate {i + 1}/{len(population)}...")
            candidate = evaluate_candidate(self.llm, candidate, test_suite)
            population[i] = candidate

        score_population(population, config)

    def _save_generation(
        self, population: list[Candidate], generation: int
    ) -> None:
        """Save all candidates in a generation to storage."""
        for candidate in population:
            candidate.generation = generation
            self.storage.save_candidate(candidate, self.experiment.id)

    def _check_convergence(self, config: ExperimentConfig) -> bool:
        """Check if the top scores have plateaued."""
        if len(self._convergence_history) < config.plateau_rounds + 1:
            return False

        recent = self._convergence_history[-config.plateau_rounds:]
        if max(recent) - min(recent) <= config.plateau_threshold:
            return True
        return False

    def _print_gen_summary(self, gen: int, population: list[Candidate]) -> None:
        """Print a summary of the current generation."""
        best = find_best_candidate(population)
        if not best:
            return
        avg = sum(c.composite_score for c in population) / len(population)
        console.print(
            f"  [yellow]Best:[/] {best.composite_score:.4f}  "
            f"[dim]Avg: {avg:.4f}  Pop: {len(population)}[/]"
        )

    def _print_final_summary(self) -> None:
        """Print the final experiment summary."""
        best = find_best_candidate(self.experiment.winners)
        stats = self.storage.get_experiment_stats(self.experiment.id)

        # Compute duration
        duration = self.experiment.updated_at - self.experiment.created_at
        duration_secs = duration.total_seconds()
        if duration_secs >= 3600:
            duration_str = f"{duration_secs / 3600:.1f}h"
        elif duration_secs >= 60:
            duration_str = f"{duration_secs / 60:.1f}m"
        else:
            duration_str = f"{duration_secs:.0f}s"

        table = Table(title="Experiment Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Status", self.experiment.status.value)
        table.add_row("Generations", str(self.experiment.current_generation))
        table.add_row("Duration", duration_str)
        table.add_row(
            "Total Cost",
            f"${stats['total_cost']:.6f}"
        )
        table.add_row(
            "Total Tokens",
            f"{stats['total_tokens']:,}"
        )
        table.add_row(
            "LLM Calls",
            str(stats["total_candidates"])
        )
        if best:
            table.add_row("Best Score", f"{best.composite_score:.4f}")
            table.add_row("Best System Prompt", best.prompts.system_prompt[:80] + "...")
            table.add_row(
                "Best User Prompt", best.prompts.user_prompt[:80] + "..."
            )
        console.print(table)
