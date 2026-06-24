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
from .kpi import find_best_candidate
from .llm import DeepSeekClient
from .models import Experiment, ExperimentConfig, TestCase, TestSuite
from .self_optimizer import SelfOptimizationRunner
from .storage import Storage
from .test_generator import generate_test_suite
from .utils import resolve_system_prompt

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
    system_prompt: str = typer.Option(
        "", "--system-prompt", "-s",
        help="System prompt (inline text or path to a .txt file). "
             "If the value is an existing file path, its content is read. "
             "Otherwise the value is used directly as inline text.",
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
        evaluation_model=data.get("evaluation_model", "deepseek-v4-flash"),
    )

    if not test_suite.test_cases:
        console.print("[red]✗ Test suite must have at least one test case[/]")
        raise typer.Exit(code=1)

    # Resolve system prompt: supports both file paths and inline text
    resolved_system_prompt = resolve_system_prompt(system_prompt) if system_prompt else ""

    exp_name = name or f"Experiment: {task[:40]}"
    experiment = Experiment(
        name=exp_name,
        task_description=task,
        test_suite=test_suite,
    )

    if resolved_system_prompt:
        experiment.config.system_prompt = resolved_system_prompt
        console.print(f"  System prompt: {resolved_system_prompt[:60]}...")

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
        "deepseek-v4-flash", "--model", "-m", help="DeepSeek model name"
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
    stats = storage.get_experiment_stats(experiment_id)

    # Experiment-level stats table
    duration = experiment.updated_at - experiment.created_at
    duration_secs = duration.total_seconds()
    if duration_secs >= 3600:
        duration_str = f"{duration_secs / 3600:.1f}h"
    elif duration_secs >= 60:
        duration_str = f"{duration_secs / 60:.1f}m"
    else:
        duration_str = f"{duration_secs:.0f}s"

    stats_table = Table(title="Experiment Stats")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Status", experiment.status.value)
    stats_table.add_row("Generations", str(experiment.current_generation))
    stats_table.add_row("Duration", duration_str)
    stats_table.add_row("Total Cost", f"${stats['total_cost']:.6f}")
    stats_table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
    stats_table.add_row("LLM Calls", str(stats["total_candidates"]))
    stats_table.add_row("Avg Cost/Call", f"${stats['avg_cost']:.6f}")
    console.print(stats_table)

    if winner_only and winners:
        best = find_best_candidate(winners)
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


@app.command()
def generate_tests(
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="System prompt text or path to a file containing it"
    ),
    user_prompt: str = typer.Option(
        ..., "--user-prompt", "-u", help="User prompt text or path to a file containing it"
    ),
    task: str = typer.Option(
        ..., "--task", "-t", help="Task description for the experiment"
    ),
    output: str = typer.Option(
        "tests.json", "--output", "-o", help="Output path for tests.json"
    ),
    count: int = typer.Option(
        5, "--count", "-c", help="Number of test cases to generate", min=1, max=20
    ),
    model: str = typer.Option(
        "deepseek-v4-flash", "--model", "-m", help="DeepSeek model for test generation"
    ),
):
    """Generate a tests.json file from existing prompts using LLM.

    Uses the LLM to analyze your existing prompts and generate diverse
    test cases (inputs + rubrics) that can be used with 'abx init'.
    """
    # Resolve prompts — allow reading from files, fallback to inline text
    # Uses resolve_system_prompt() which safely handles OSError from long paths
    system_prompt_text = resolve_system_prompt(system_prompt)
    user_prompt_text = resolve_system_prompt(user_prompt)

    if not system_prompt_text and not user_prompt_text:
        console.print("[red]✗ At least one of --system-prompt or --user-prompt must be non-empty[/]")
        raise typer.Exit(code=1)

    # Initialize LLM client
    try:
        llm = DeepSeekClient(model=model)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/]")
        console.print("  Set DEEPSEEK_API_KEY environment variable or ensure it's configured.")
        raise typer.Exit(code=1)

    console.print("[bold cyan]Generating test cases from prompts...[/]")
    console.print(f"  Task: {task}")
    console.print(f"  Test cases to generate: {count}")
    console.print(f"  System prompt: {system_prompt_text[:60]}...")
    console.print(f"  User prompt: {user_prompt_text[:60]}...")

    try:
        test_suite = generate_test_suite(
            llm_client=llm,
            task_description=task,
            system_prompt=system_prompt_text,
            user_prompt=user_prompt_text,
            count=count,
            model=model,
        )
    except ValueError as e:
        console.print(f"[red]✗ Failed to generate test suite: {e}[/]")
        raise typer.Exit(code=1)

    # Write output
    output_path = Path(output)
    output_data = test_suite.model_dump(mode="json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[green]✓[/] Generated [bold]{len(test_suite.test_cases)}[/] test cases")
    console.print(f"  Output: [bold]{output_path.resolve()}[/]")
    console.print(f"\n  Use with: [bold]abx init --task \"{task}\" --tests {output}[/]")


@app.command()
def self_optimize(
    tests: str = typer.Option(
        ..., "--tests", "-f", help="Path to test suite JSON file with expected_score fields"
    ),
    task: str = typer.Option(
        "", "--task", "-t", help="Task description (overrides test suite value)"
    ),
    name: str = typer.Option(
        "Experiment A: Optimize EVAL_SYSTEM_PROMPT via GA", "--name", "-n",
        help="Experiment name",
    ),
    db: str = typer.Option(
        "ab_explorer.db", "--db", "-d", help="SQLite database path"
    ),
    cycles: int = typer.Option(
        5, "--cycles", "-c", help="Maximum optimization generations"
    ),
    population_size: int = typer.Option(
        8, "--population", "-p", help="Population size per generation"
    ),
    model: str = typer.Option(
        "deepseek-v4-flash", "--model", "-m", help="DeepSeek model name"
    ),
    accuracy_weight: float = typer.Option(
        0.6, "--accuracy-weight", help="KPI accuracy weight (primary)"
    ),
    cost_weight: float = typer.Option(
        0.2, "--cost-weight", help="KPI cost weight"
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
    """Run GA self-optimization of EVAL_SYSTEM_PROMPT.

    Uses a test suite with expected_score fields to optimize the
    evaluator's system prompt for accuracy via genetic algorithm.
    """
    # Load test suite
    tests_path = Path(tests)
    if not tests_path.exists():
        console.print(f"[red]✗ Test file not found: {tests}[/]")
        raise typer.Exit(code=1)

    with open(tests_path) as f:
        data = json.load(f)

    test_suite = TestSuite(
        task_description=data.get("task_description", task or "Evaluate AI responses against rubrics"),
        test_cases=[TestCase(**tc) for tc in data.get("test_cases", [])],
        evaluation_model=data.get("evaluation_model", "deepseek-v4-flash"),
    )

    if not test_suite.test_cases:
        console.print("[red]✗ Test suite must have at least one test case[/]")
        raise typer.Exit(code=1)

    # Verify test cases have expected_score
    missing_scores = [tc for tc in test_suite.test_cases if tc.expected_score is None]
    if missing_scores:
        console.print(f"[yellow]⚠ {len(missing_scores)}/{len(test_suite.test_cases)} test cases missing expected_score[/]")
        console.print("  These cases will use default score of 5.0.")

    experiment = Experiment(
        name=name,
        task_description=test_suite.task_description,
        test_suite=test_suite,
    )

    # Set config
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

    # Save experiment
    storage = Storage(db_path=db)
    storage.save_experiment(experiment)

    console.print(f"[green]✓[/] Self-optimization experiment initialized: [bold]{experiment.id}[/]")
    console.print(f"  Name: {experiment.name}")
    console.print(f"  Generations: {cycles} | Population: {population_size}")
    console.print(f"  Test cases: {len(test_suite.test_cases)}")
    console.print(f"  Database: {db}")
    console.print(f"\n  Running optimization...\n")

    # Run self-optimization
    runner = SelfOptimizationRunner(experiment, storage, llm)
    result = runner.run()

    # Save final state
    storage.save_experiment(result)

    console.print(f"\n[green]✓[/] Self-optimization complete!")
    console.print(f"  ID: {result.id}")
    console.print(f"  Status: {result.status.value}")
    console.print(f"  Generations: {result.current_generation}")

    # Offer baseline comparison
    console.print(f"\n  Run comparison: [bold]abx self-eval-compare --experiment-id {result.id} --db {db}[/]")


@app.command()
def self_eval_compare(
    experiment_id: str = typer.Option(
        ..., "--experiment-id", "-e", help="Self-optimization experiment ID"
    ),
    db: str = typer.Option(
        "ab_explorer.db", "--db", "-d", help="SQLite database path"
    ),
    model: str = typer.Option(
        "deepseek-v4-flash", "--model", "-m", help="DeepSeek model name"
    ),
):
    """Compare the winning EVAL_SYSTEM_PROMPT against the baseline."""
    storage = Storage(db_path=db)
    experiment = storage.get_experiment(experiment_id)

    if not experiment:
        console.print(f"[red]✗ Experiment not found: {experiment_id}[/]")
        raise typer.Exit(code=1)

    if not experiment.test_suite:
        console.print("[red]✗ Experiment has no test suite[/]")
        raise typer.Exit(code=1)

    # Initialize LLM
    try:
        llm = DeepSeekClient(model=model)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/]")
        raise typer.Exit(code=1)

    # Get winning candidate
    winners = storage.get_winners(experiment_id)
    if not winners:
        console.print("[yellow]⚠ No winners found. Run self-optimize first.[/]")
        return

    best = max(winners, key=lambda c: c.composite_score)

    from .self_optimizer import evaluate_eval_candidate, compute_selfopt_composite_score
    from .evaluator import EVAL_SYSTEM_PROMPT as BASELINE_EVAL_PROMPT
    from .models import Candidate, PromptPair

    console.print("[bold cyan]=== Baseline EVAL_SYSTEM_PROMPT ===[/]")
    console.print(BASELINE_EVAL_PROMPT)
    console.print()

    # Evaluate baseline
    console.print("[bold]Evaluating baseline prompt...[/]")
    baseline_candidate = Candidate(
        prompts=PromptPair(system_prompt=BASELINE_EVAL_PROMPT, user_prompt=""),
        generation=-1,
        mutation_type="baseline",
    )
    baseline_candidate, baseline_accuracy = evaluate_eval_candidate(
        llm, baseline_candidate, experiment.test_suite
    )

    console.print(f"  Baseline accuracy: {baseline_accuracy:.2%}")
    console.print(f"  Baseline cost: ${baseline_candidate.cost:.6f}")

    console.print("\n[bold cyan]=== Winning EVAL_SYSTEM_PROMPT ===[/]")
    console.print(best.prompts.system_prompt)
    console.print()

    # Evaluate winning prompt
    console.print("[bold]Evaluating winning prompt...[/]")
    best_accuracy = best.avg_score() if best.scores else 0.0

    console.print(f"  Winning accuracy: {best_accuracy:.2%}")
    console.print(f"  Winning cost: ${best.cost:.6f}")

    # Comparison table
    comparison = Table(title="Comparison: Baseline vs Optimized EVAL_SYSTEM_PROMPT")
    comparison.add_column("Metric", style="cyan")
    comparison.add_column("Baseline", style="yellow")
    comparison.add_column("Optimized", style="green")
    comparison.add_column("Delta", style="blue")

    acc_delta = best_accuracy - baseline_accuracy
    cost_delta = best.cost - baseline_candidate.cost
    lat_delta = best.latency - baseline_candidate.latency

    comparison.add_row(
        "Accuracy",
        f"{baseline_accuracy:.2%}",
        f"{best_accuracy:.2%}",
        f"{acc_delta:+.2%}",
    )
    comparison.add_row(
        "Cost",
        f"${baseline_candidate.cost:.6f}",
        f"${best.cost:.6f}",
        f"${cost_delta:+.6f}",
    )
    comparison.add_row(
        "Avg Latency",
        f"{baseline_candidate.latency:.0f}ms",
        f"{best.latency:.0f}ms",
        f"{lat_delta:+.0f}ms",
    )
    comparison.add_row(
        "Test Cases",
        str(len(experiment.test_suite.test_cases)),
        str(len(experiment.test_suite.test_cases)),
        "—",
    )

    console.print(comparison)

    # Improvement assessment
    if acc_delta > 0.05:
        console.print(f"\n[bold green]✓ Significant improvement: Accuracy +{acc_delta:.2%}[/]")
    elif acc_delta > 0:
        console.print(f"\n[green]✓ Marginal improvement: Accuracy +{acc_delta:.2%}[/]")
    elif acc_delta == 0:
        console.print(f"\n[yellow]— No change in accuracy[/]")
    else:
        console.print(f"\n[red]✗ Accuracy decreased by {abs(acc_delta):.2%}[/]")


def main():
    """Entry point for installed package."""
    app()


if __name__ == "__main__":
    main()
