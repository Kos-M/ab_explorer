"""Self-optimization module — uses GA to optimize the EVAL_SYSTEM_PROMPT.

This is a meta-optimization: instead of optimizing candidate prompts that
answer test questions, we optimize the evaluator's own system prompt to
produce more accurate scores against ground-truth expected scores.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime

from rich.console import Console
from rich.table import Table

from .evaluator import EVAL_USER_PROMPT, evaluate_candidate
from .llm import DeepSeekClient, LLMResponse
from .models import (
    Candidate,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    PromptPair,
    TestCase,
    TestSuite,
)
from .population import (
    CROSSOVER_PROMPT,
    MUTATION_PROMPT,
    evolve_population as _evolve_population_base,
    generate_initial_population as _gen_initial_base,
    tournament_select,
)
from .storage import Storage

console = Console()


# Prompt for generating initial EVAL_SYSTEM_PROMPT variants
INITIAL_EVAL_PROMPT = """You are a prompt engineering expert. Generate {count} different evaluation system prompts.

An evaluation system prompt instructs an LLM to score AI responses against a rubric on a scale of 0-10.
Each variant should take a DIFFERENT approach to evaluating responses.

Current (baseline) evaluation system prompt:
"{baseline_prompt}"

Task for which these prompts will evaluate responses:
"{task_description}"

For EACH variant, provide ONLY a system prompt — a detailed instruction for how the LLM should
evaluate responses. Vary the following dimensions across variants:
- Strictness vs leniency
- Level of analytical detail requested
- Specific scoring guidelines vs general principles
- Whether to use chain-of-thought reasoning before scoring
- Emphasis on different aspects (factual accuracy, reasoning quality, completeness, etc.)

Return your response as a JSON array of objects, each with ONLY a "system_prompt" key.
No markdown, no code blocks — just raw JSON."""

# Modified mutation prompt for eval prompts
EVAL_MUTATION_PROMPT = """You are a prompt engineer improving an evaluation system prompt.
The evaluation prompt is used to score AI responses against a rubric on a scale of 0-10.

Current evaluation system prompt: {system_prompt}

Task being evaluated: {task_description}

Improve this evaluation prompt to produce MORE ACCURATE scores. You can:
- Make scoring criteria more specific
- Add calibration examples
- Improve rubrics interpretation guidelines
- Add chain-of-thought reasoning steps
- Adjust strictness/leniency calibration
- Improve consistency across different response types

Return your response as a JSON object with a "system_prompt" key.
No markdown, no code blocks — just raw JSON."""

# Modified crossover prompt for eval prompts
EVAL_CROSSOVER_PROMPT = """You are a prompt engineer combining the best parts of two parent evaluation prompts into a new child prompt.

Evaluation prompts score AI responses against rubrics on a scale of 0-10.

Parent A: {sys_a}
Parent B: {sys_b}

Task being evaluated: {task_description}

Create a new evaluation prompt that takes the best elements from both parents.
The result should be a coherent, improved evaluation prompt that produces accurate scores.

Return your response as a JSON object with a "system_prompt" key.
No markdown, no code blocks — just raw JSON."""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("json"):
        text = text[4:]
    return text.strip()


def _parse_json_response(text: str):
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]|\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def generate_initial_eval_population(
    llm_client: DeepSeekClient,
    task_description: str,
    population_size: int = 8,
    baseline_prompt: str = "",
    temperature: float = 0.8,
) -> list[Candidate]:
    """Generate initial population of EVAL_SYSTEM_PROMPT variants."""
    baseline = baseline_prompt or "You are an expert evaluator scoring AI responses against a rubric."

    prompt = INITIAL_EVAL_PROMPT.format(
        count=population_size,
        baseline_prompt=baseline,
        task_description=task_description,
    )

    response = llm_client.chat(
        system_prompt="You are a JSON-generating prompt engineering expert. Always return valid JSON.",
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=4096,
    )

    parsed = _parse_json_response(response.content)
    candidates = []

    if isinstance(parsed, list):
        for item in parsed[:population_size]:
            if isinstance(item, dict) and item.get("system_prompt"):
                candidates.append(Candidate(
                    prompts=PromptPair(
                        system_prompt=item["system_prompt"],
                        user_prompt="",
                    ),
                    generation=0,
                ))
    elif isinstance(parsed, dict) and parsed.get("system_prompt"):
        candidates.append(Candidate(
            prompts=PromptPair(
                system_prompt=parsed["system_prompt"],
                user_prompt="",
            ),
            generation=0,
        ))

    # Ensure we have enough by repeating/padding with baseline variants
    while len(candidates) < population_size:
        candidates.append(Candidate(
            prompts=PromptPair(
                system_prompt=baseline,
                user_prompt="",
            ),
            generation=0,
        ))

    return candidates


def evaluate_eval_prompt(
    llm_client: DeepSeekClient,
    eval_system_prompt: str,
    test_case: TestCase,
    task_description: str,
) -> dict:
    """Evaluate how well an EVAL_SYSTEM_PROMPT variant scores a single test case.

    The test_case.input is treated as the AI response to evaluate,
    and test_case.expected_score is the ground truth.

    Returns dict with: score, reasoning, cost, latency, token_count, accuracy.
    """
    score_response = llm_client.chat(
        system_prompt=eval_system_prompt,
        user_prompt=EVAL_USER_PROMPT.format(
            task_description=task_description,
            response=test_case.input,
            rubric=test_case.rubric,
        ),
        temperature=0.2,
        max_tokens=512,
    )

    # Parse score from evaluator response
    score = 5.0
    try:
        parsed = json.loads(score_response.content)
        if isinstance(parsed, dict):
            score = float(parsed.get("score", 5))
        elif isinstance(parsed, (int, float)):
            score = float(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score = max(0.0, min(10.0, score))

    # Compute accuracy vs expected score
    expected = test_case.expected_score or 5.0
    error = abs(score - expected)
    accuracy = max(0.0, 1.0 - error / 10.0)

    return {
        "score": score,
        "expected": expected,
        "accuracy": round(accuracy, 4),
        "error": round(error, 2),
        "reasoning": score_response.content[:200],
        "cost": score_response.cost,
        "latency": score_response.latency_ms,
        "token_count": score_response.total_tokens,
    }


def evaluate_eval_candidate(
    llm_client: DeepSeekClient,
    candidate: Candidate,
    test_suite: TestSuite,
) -> Candidate:
    """Evaluate an EVAL_SYSTEM_PROMPT variant against all test cases.

    Scores the candidate on:
    - Accuracy: how close scores are to expected scores
    - Cost: total LLM cost
    - Latency: average latency
    """
    eval_prompt = candidate.prompts.system_prompt
    task_desc = test_suite.task_description

    total_cost = 0.0
    total_latency = 0.0
    total_tokens = 0
    scores = []
    accuracies = []

    for test_case in test_suite.test_cases:
        result = evaluate_eval_prompt(
            llm_client, eval_prompt, test_case, task_desc
        )
        scores.append(result["score"])
        accuracies.append(result["accuracy"])
        total_cost += result["cost"]
        total_latency += result["latency"]
        total_tokens += result["token_count"]

    candidate.scores = accuracies  # Store accuracy per test case as scores
    candidate.cost = total_cost
    candidate.latency = total_latency / len(test_suite.test_cases) if test_suite.test_cases else 0.0
    candidate.token_count = total_tokens

    # Primary KPI: mean accuracy (0-1 scale)
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return candidate, mean_accuracy


def compute_selfopt_composite_score(
    candidate: Candidate,
    config: ExperimentConfig,
    baseline_accuracy: float = 0.5,
    baseline_cost: float = 0.001,
    baseline_latency: float = 1000.0,
) -> float:
    """Compute composite score for self-optimization candidates.

    Primary: accuracy (weighted heavily)
    Secondary: cost and latency
    """
    weights = config.kpi_weights
    w_a = weights.get("accuracy", 0.6)
    w_c = weights.get("cost", 0.2)
    w_l = weights.get("latency", 0.2)

    # Accuracy is the mean of per-test-case accuracies stored in scores
    accuracy = candidate.avg_score() if candidate.scores else 0.0

    # Relative cost (lower is better)
    rel_cost = candidate.cost / max(baseline_cost, candidate.cost, 0.0001)
    cost_factor = 1.0 - min(rel_cost, 1.0)

    # Relative latency (lower is better)
    rel_latency = candidate.latency / max(baseline_latency, candidate.latency, 1.0)
    latency_factor = 1.0 - min(rel_latency, 1.0)

    score = (w_a * accuracy) + (w_c * cost_factor) + (w_l * latency_factor)
    return round(score, 4)


def mutate_eval_prompt(
    llm_client: DeepSeekClient,
    candidate: Candidate,
    task_description: str,
    mutation_rate: float = 0.3,
) -> Candidate:
    """Mutate an EVAL_SYSTEM_PROMPT using the LLM."""
    if random.random() > mutation_rate:
        return Candidate(
            prompts=candidate.prompts,
            parent_id=candidate.id,
            mutation_type="no_mutation",
        )

    prompt = EVAL_MUTATION_PROMPT.format(
        system_prompt=candidate.prompts.system_prompt,
        task_description=task_description,
    )

    try:
        response = llm_client.chat(
            system_prompt="You are a prompt engineering expert. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.9,
            max_tokens=2048,
        )
        parsed = _parse_json_response(response.content)
        if parsed and isinstance(parsed, dict) and parsed.get("system_prompt"):
            return Candidate(
                prompts=PromptPair(
                    system_prompt=parsed["system_prompt"],
                    user_prompt="",
                ),
                parent_id=candidate.id,
                mutation_type="llm_mutation",
            )
    except Exception:
        pass

    return Candidate(
        prompts=candidate.prompts,
        parent_id=candidate.id,
        mutation_type="no_mutation",
    )


def crossover_eval_prompts(
    parent_a: Candidate,
    parent_b: Candidate,
    task_description: str,
    llm_client: DeepSeekClient = None,
) -> Candidate:
    """Create a child eval prompt by combining two parents."""
    if llm_client:
        prompt = EVAL_CROSSOVER_PROMPT.format(
            sys_a=parent_a.prompts.system_prompt,
            sys_b=parent_b.prompts.system_prompt,
            task_description=task_description,
        )
        try:
            response = llm_client.chat(
                system_prompt="You are a prompt engineering expert. Return only valid JSON.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=2048,
            )
            parsed = _parse_json_response(response.content)
            if parsed and isinstance(parsed, dict) and parsed.get("system_prompt"):
                return Candidate(
                    prompts=PromptPair(
                        system_prompt=parsed["system_prompt"],
                        user_prompt="",
                    ),
                    parent_id=parent_a.id,
                    mutation_type="crossover_llm",
                )
        except Exception:
            pass

    # Simple crossover: take parent A's prompt as-is
    return Candidate(
        prompts=PromptPair(
            system_prompt=parent_a.prompts.system_prompt,
            user_prompt="",
        ),
        parent_id=parent_a.id,
        mutation_type="crossover_swap",
    )


def evolve_eval_population(
    llm_client: DeepSeekClient,
    population: list[Candidate],
    task_description: str,
    config: ExperimentConfig,
) -> list[Candidate]:
    """Create the next generation of EVAL_SYSTEM_PROMPT variants."""
    next_generation = []
    next_gen_number = max(c.generation for c in population) + 1

    # Elitism: keep the best candidate
    best = max(population, key=lambda c: c.composite_score)
    next_generation.append(Candidate(
        prompts=best.prompts,
        generation=next_gen_number,
        parent_id=best.id,
        mutation_type="elite",
    ))

    # Generate rest of population
    while len(next_generation) < config.population_size:
        if random.random() < config.crossover_rate and len(population) >= 2:
            parent_a = tournament_select(population, config.tournament_size)
            parent_b = tournament_select(population, config.tournament_size)
            child = crossover_eval_prompts(parent_a, parent_b, task_description, llm_client)
        else:
            parent = tournament_select(population, config.tournament_size)
            child = mutate_eval_prompt(llm_client, parent, task_description, config.mutation_rate)

        child.generation = next_gen_number
        next_generation.append(child)

    return next_generation


class SelfOptimizationRunner:
    """Runs GA optimization of the EVAL_SYSTEM_PROMPT."""

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
        """Execute the full self-optimization loop."""
        self.experiment.status = ExperimentStatus.RUNNING
        self.experiment.updated_at = datetime.now()
        self.storage.save_experiment(self.experiment)

        config = self.experiment.config
        test_suite = self.experiment.test_suite
        if not test_suite:
            raise ValueError("Experiment has no test suite configured")

        console.print(f"\n[bold magenta]🔬 Starting self-optimization:[/] {self.experiment.name}")
        console.print(f"[dim]Generations: {config.cycles} | Population: {config.population_size}[/]")
        console.print(f"[dim]Test cases: {len(test_suite.test_cases)} | Accuracy weight: {config.kpi_weights.get('accuracy', 0.6)}[/]\n")

        # Generation 0: initial population
        console.print("[bold]Generation 0:[/] Generating initial EVAL_SYSTEM_PROMPT variants...")
        from .evaluator import EVAL_SYSTEM_PROMPT as BASELINE_EVAL_PROMPT
        population = generate_initial_eval_population(
            self.llm,
            test_suite.task_description,
            config.population_size,
            baseline_prompt=BASELINE_EVAL_PROMPT,
        )

        self._evaluate_and_score(population, test_suite, config)
        self._save_generation(population, 0)
        self._print_gen_summary(0, population)

        # Evolve through generations
        for gen in range(1, config.cycles + 1):
            console.print(f"[bold]Generation {gen}:[/] Evolving...")
            population = evolve_eval_population(
                self.llm, population, test_suite.task_description, config
            )
            self._evaluate_and_score(population, test_suite, config)
            self._save_generation(population, gen)
            self._print_gen_summary(gen, population)

            # Track best accuracy for convergence
            best = max(population, key=lambda c: c.composite_score)
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
        # Find baseline for normalization
        first_results = []
        for i, candidate in enumerate(population):
            console.print(f"  Evaluating candidate {i + 1}/{len(population)}...")
            candidate, mean_acc = evaluate_eval_candidate(self.llm, candidate, test_suite)
            population[i] = candidate
            first_results.append(mean_acc)

        # Score population using relative accuracy/cost/latency
        baseline_cost = max(c.cost for c in population) if population else 0.001
        baseline_latency = max(c.latency for c in population) if population else 1000.0

        for candidate in population:
            candidate.composite_score = compute_selfopt_composite_score(
                candidate, config, baseline_cost=baseline_cost, baseline_latency=baseline_latency
            )

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
        best = max(population, key=lambda c: c.composite_score)
        if not best:
            return
        avg = sum(c.composite_score for c in population) / len(population)
        best_accuracy = best.avg_score() if best.scores else 0.0
        console.print(
            f"  [yellow]Best:[/] {best.composite_score:.4f} "
            f"[dim](accuracy: {best_accuracy:.2%}) Avg: {avg:.4f} Pop: {len(population)}[/]"
        )

    def _print_final_summary(self) -> None:
        """Print the final experiment summary."""
        best = max(self.experiment.winners, key=lambda c: c.composite_score) if self.experiment.winners else None
        stats = self.storage.get_experiment_stats(self.experiment.id)

        duration = self.experiment.updated_at - self.experiment.created_at
        duration_secs = duration.total_seconds()
        if duration_secs >= 3600:
            duration_str = f"{duration_secs / 3600:.1f}h"
        elif duration_secs >= 60:
            duration_str = f"{duration_secs / 60:.1f}m"
        else:
            duration_str = f"{duration_secs:.0f}s"

        # Load baseline for comparison
        from .evaluator import EVAL_SYSTEM_PROMPT as BASELINE_EVAL_PROMPT

        table = Table(title="Self-Optimization Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Status", self.experiment.status.value)
        table.add_row("Generations", str(self.experiment.current_generation))
        table.add_row("Duration", duration_str)
        table.add_row("Total Cost", f"${stats['total_cost']:.6f}")
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        table.add_row("LLM Calls", str(stats["total_candidates"]))
        if best:
            best_accuracy = best.avg_score() if best.scores else 0.0
            table.add_row("Best Composite", f"{best.composite_score:.4f}")
            table.add_row("Best Accuracy", f"{best_accuracy:.2%}")
            table.add_row("Best Cost", f"${best.cost:.6f}")
        console.print(table)

        if best:
            console.print("\n[bold cyan]=== Winning EVAL_SYSTEM_PROMPT ===[/]")
            console.print(best.prompts.system_prompt)

            console.print("\n[bold cyan]=== Baseline EVAL_SYSTEM_PROMPT ===[/]")
            console.print(BASELINE_EVAL_PROMPT)

            # Compare
            console.print("\n[bold yellow]Comparison:[/]")
            console.print(f"  Baseline accuracy: N/A (will be evaluated separately)")
            console.print(f"  Winning accuracy:  {best.avg_score():.2%}" if best.scores else "  N/A")

        console.print(f"\n[bold]Next step:[/] Run comparison test:")
        console.print(f"  abx self-eval-compare --experiment-id {self.experiment.id} --db {self.storage.db_path}")

    def evaluate_baseline(self) -> dict:
        """Evaluate the baseline EVAL_SYSTEM_PROMPT against the test suite.

        Returns dict with accuracy, cost, latency metrics for comparison.
        """
        from .evaluator import EVAL_SYSTEM_PROMPT as BASELINE_EVAL_PROMPT
        test_suite = self.experiment.test_suite
        if not test_suite:
            raise ValueError("No test suite configured")

        baseline_candidate = Candidate(
            prompts=PromptPair(
                system_prompt=BASELINE_EVAL_PROMPT,
                user_prompt="",
            ),
            generation=-1,  # Special marker for baseline
            mutation_type="baseline",
        )

        baseline_candidate, mean_acc = evaluate_eval_candidate(
            self.llm, baseline_candidate, test_suite
        )

        return {
            "accuracy": mean_acc,
            "cost": baseline_candidate.cost,
            "latency": baseline_candidate.latency,
            "token_count": baseline_candidate.token_count,
            "scores": baseline_candidate.scores,
        }
