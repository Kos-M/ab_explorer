"""CLI entry point for ab_explorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .experiment import ExperimentRunner
from .llm import DeepSeekClient
from .models import Experiment, ExperimentConfig, TestCase, TestSuite
from .storage import Storage

app = typer.Typer(
    name="abx",
    help="A/B testing CLI tool for prompt optimization using genetic algorithms",
)
console = Console()


@app.callback()
def callback():
    """ab_explorer - Prompt A/B Testing CLI"""
    pass


@app.command()
def init(
    task: str = typer.Option(..., "--task", "-t", help="Task description for the experiment"),
    tests: str = typer.Option(
        ..., "--tests", "-f", help="Path to test suite JSON file"
    ),
    name: str = typer.Option("", "--name", "-n", help="Optional experiment name"),
    output: str = typer.Option(
        "ab_explorer.db", "--output", "-o", help="SQLite database path"
    ),
):
    """Initialize a new experiment with a test suite."""
    # Load test suite
    tests_path = Path(tests)
    if not tests_path.exists():
        console.print(f"[red]✗ Test file not found: {tests}[/]")
        raise typer.Exit(code=1)

    with open(tests_path) as f:
        data = json.load(f)

    test_suite = TestSuite(
        task_description=data.get("task_description", task),
        test_cases=[TestCase(**tc) for tc in data.get("test_cases", [])],
        evaluation_model=data.get("evaluation_model", "deepseek-chat"),
    )

    if not test_suite.test_cases:
        console.print("[red]✗ Test suite must have at least one test case[/]")
        raise typer.Exit(code=1)

    exp_name = name or f"Experiment: {task[:40]}"
    experiment = Experiment(
        name=exp_name,
        task_description=task,
        test_suite=test_suite,
    )

    storage = Storage(db_path=output)
    storage.save_experiment(experiment)

    console.print(f"[green]✓[/] Experiment initialized: [bold]{experiment.id}[/]")
    console.print(f"  Name: {experiment.name}")
    console.print(f"  Task: {task}")
    console.print(f"  Test cases: {len(test_suite.test_cases)}")
    console.print(f"  Database: {output}")
    console.print(f"\n  Run: [bold]abx run --experiment-id {experiment.id}[/]")


@app.command()
def run(
    experiment_id: str = typer.Option(
        ..., "--experiment-id", "-e", help="Experiment ID to run"
    ),
    cycles: int = typer.Option(
        20, "--cycles", "-c", help="Maximum optimization cycles"
    ),
    population_size: int = typer.Option(
        5, "--population", "-p", help="Population size per generation"
    ),
    model: str = typer.Option(
        "deepseek-chat", "--model", "-m", help="DeepSeek model name"
    ),
    db: str = typer.Option(
        "ab_explorer.db", "--db", "-d", help="SQLite database path"
    ),
    accuracy_weight: float = typer.Option(
        0.5, "--accuracy-weight", help="KPI accuracy weight"
    ),
    cost_weight: float = typer.Option(
        0.3, "--cost-weight", help="KPI cost weight"
    ),
    latency_weight: float = typer.Option(
        0.2, "--latency-weight", help="KPI latency weight"
    ),
    plateau_threshold: float = typer.Option(
        0.02, "--plateau-threshold", help="Convergence threshold"
    ),
    plateau_rounds: int = typer.Option(
        5, "--plateau-rounds", help="Rounds before convergence"
    ),
):
    """Run the optimization loop for an experiment."""
    storage = Storage(db_path=db)
    experiment = storage.get_experiment(experiment_id)

    if not experiment:
        console.print(f"[red]✗ Experiment not found: {experiment_id}[/]")
        raise typer.Exit(code=1)

    # Override config with CLI args
    experiment.config.cycles = cycles
    experiment.config.population_size = population_size
    experiment.config.model = model
    experiment.config.plateau_threshold = plateau_threshold
    experiment.config.plateau_rounds = plateau_rounds
    experiment.config.kpi_weights = {
        "accuracy": accuracy_weight,
        "cost": cost_weight,
        "latency": latency_weight,
    }

    # Initialize LLM client
    try:
        llm = DeepSeekClient(model=model)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/]")
        console.print("  Set DEEPSEEK_API_KEY environment variable or ensure it's configured.")
        raise typer.Exit(code=1)

    # Run experiment
    runner = ExperimentRunner(experiment, storage, llm)
    result = runner.run()

    # Save final state
    storage.save_experiment(result)

    console.print(f"\n[green]✓[/] Experiment complete!")
    console.print(f"  ID: {result.id}")
    console.print(f"  Status: {result.status.value}")
    console.print(f"  Generations: {result.current_generation}")


@app.command()
def report(
    experiment_id: str = typer.Option(
        ..., "--experiment-id", "-e", help="Experiment ID"
    ),
    db: str = typer.Option(
        "ab_explorer.db", "--db", "-d", help="SQLite database path"
    ),
    winner_only: bool = typer.Option(
        False, "--winner-only", "-w", help="Show only the winning prompt"
    ),
):
    """View experiment results."""
    storage = Storage(db_path=db)
    experiment = storage.get_experiment(experiment_id)

    if not experiment:
        console.print(f"[red]✗ Experiment not found: {experiment_id}[/]")
        raise typer.Exit(code=1)

    winners = storage.get_winners(experiment_id)
    candidates = storage.get_candidates(experiment_id)

    if winner_only and winners:
        best = winners[0]
        console.print("\n[bold cyan]=== Winning Prompt ===[/]")
        console.print(f"\n[bold]System Prompt:[/]\n{best.prompts.system_prompt}")
        console.print(f"\n[bold]User Prompt:[/]\n{best.prompts.user_prompt}")
        console.print(f"\n[bold]Score:[/] {best.composite_score:.4f}")
        console.print(f"[bold]Cost:[/] ${best.cost:.6f}")
        console.print(f"[bold]Latency:[/] {best.latency:.0f}ms")
        return

    # Full report
    table = Table(title=f"Experiment: {experiment.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ID", experiment.id)
    table.add_row("Task", experiment.task_description[:60])
    table.add_row("Status", experiment.status.value)
    table.add_row("Generations", str(experiment.current_generation))
    if experiment.test_suite:
        table.add_row("Test Cases", str(len(experiment.test_suite.test_cases)))
    console.print(table)

    if winners:
        gen_table = Table(title="Generation Winners")
        gen_table.add_column("Generation", style="cyan")
        gen_table.add_column("Score", style="green")
        gen_table.add_column("Cost", style="yellow")
        gen_table.add_column("Latency", style="blue")

        # Group by generation
        gen_scores = {}
        for w in winners:
            g = w.generation
            if g not in gen_scores or w.composite_score > gen_scores[g][0]:
                gen_scores[g] = (w.composite_score, w.cost, w.latency)

        for gen in sorted(gen_scores.keys()):
            score, cost, lat = gen_scores[gen]
            gen_table.add_row(str(gen), f"{score:.4f}", f"${cost:.6f}", f"{lat:.0f}ms")

        console.print(gen_table)


@app.command()
def list_experiments(
    db: str = typer.Option(
        "ab_explorer.db", "--db", "-d", help="SQLite database path"
    ),
):
    """List all experiments."""
    storage = Storage(db_path=db)
    experiments = storage.list_experiments()

    if not experiments:
        console.print("[yellow]No experiments found.[/]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Gen", style="blue")
    table.add_column("Created", style="dim")

    for exp in experiments:
        table.add_row(
            exp["id"][:12],
            exp["name"][:40],
            exp["status"],
            str(exp["current_generation"]),
            exp["created_at"][:19],
        )

    console.print(table)


def main():
    """Entry point for installed package."""
    app()


if __name__ == "__main__":
    main()
