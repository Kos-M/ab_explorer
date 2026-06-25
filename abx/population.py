"""Population generation and genetic algorithm operations for ab_explorer."""

from __future__ import annotations

import random
from typing import Optional

from .models import Candidate, ExperimentConfig, PromptPair


INITIAL_GENERATION_PROMPT = """You are a prompt engineering expert. Given the following task description, generate {count} different prompt strategies.

For EACH strategy, provide:
1. A system prompt (the instruction/role/guidelines for the AI)
2. A user prompt template (the specific instruction for this interaction)

The prompts should be DIFFERENT from each other in approach, style, and specificity. Vary the tone, level of detail, and techniques used.

Task: {task_description}

Return your response as a JSON array of objects, each with "system_prompt" and "user_prompt" keys. No markdown, no code blocks — just raw JSON."""

MUTATION_PROMPT = """You are a prompt engineer tasked with improving an existing prompt pair through a structured, analytical approach. Given the task description and current prompts, generate an improved version by systematically evaluating each component. Use the following breakdown: (1) Identify the core goal of {task_description} and assess if {system_prompt} and {user_prompt} align with it. (2) Analyze clarity and specificity: make instructions more precise or add step-by-step logic if needed, but avoid overcomplication. (3) Critically examine the placement and utility of any examples: add if missing for clarity, remove if confusing. (4) Adjust tone to be neutral, instructive, and professional. (5) Add explicit constraints (e.g., output format, length, style rules) only if they serve the task. (6) Restructure for logical flow: introduce a reasoning chain if the task benefits from multi-step thinking. Return your response as a JSON object with 'system_prompt' and 'user_prompt' keys. No markdown, no code blocks — just raw JSON.

Task: {task_description}

Current system prompt: {system_prompt}
Current user prompt: {user_prompt}"""

CROSSOVER_PROMPT = """You are a prompt engineer combining the best parts of two parent prompts into a new child prompt.

Task: {task_description}

Parent A system prompt: {sys_a}
Parent A user prompt: {user_a}

Parent B system prompt: {sys_b}
Parent B user prompt: {user_b}

Create a new prompt that takes the best elements from both parents. The result should be a coherent, improved prompt.

Return your response as a JSON object with "system_prompt" and "user_prompt" keys. No markdown, no code blocks — just raw JSON."""


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        # Find the first newline after ```
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


def _parse_json_response(text: str) -> Optional[dict | list]:
    """Try to parse JSON from LLM response, with fallbacks."""
    import json as json_module

    text = _strip_code_fence(text)
    # Try direct parse
    try:
        return json_module.loads(text)
    except json_module.JSONDecodeError:
        pass
    # Try finding JSON in the response
    import re
    # Look for [{...}] or {...}
    for pattern in [r"\[.*?\]", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json_module.loads(match.group())
            except json_module.JSONDecodeError:
                continue
    return None


def generate_initial_population(
    llm_client,
    task_description: str,
    population_size: int = 5,
    temperature: float = 0.8,
) -> list[Candidate]:
    """Generate initial population of prompt candidates using the LLM.

    The LLM generates N prompt strategies, each with system+user prompts.
    """
    prompt = INITIAL_GENERATION_PROMPT.format(
        count=population_size,
        task_description=task_description,
    )

    response = llm_client.chat(
        system_prompt="You are a JSON-generating prompt engineering expert. Always return valid JSON.",
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=4096,
    )

    parsed = _parse_json_response(response.content)
    if not parsed:
        # Fallback: return a single basic candidate
        return [Candidate(prompts=PromptPair(
            system_prompt="You are a helpful assistant.",
            user_prompt=task_description,
        ))]

    candidates = []
    if isinstance(parsed, dict):
        parsed = [parsed]

    for item in parsed[:population_size]:
        if isinstance(item, dict):
            candidates.append(Candidate(
                prompts=PromptPair(
                    system_prompt=item.get("system_prompt", ""),
                    user_prompt=item.get("user_prompt", ""),
                ),
                generation=0,
            ))

    # Fill remaining if we got fewer than expected
    while len(candidates) < population_size:
        candidates.append(Candidate(
            prompts=PromptPair(
                system_prompt="You are a helpful AI assistant.",
                user_prompt=f"Please complete the following task: {task_description}",
            ),
            generation=0,
        ))

    return candidates


def tournament_select(
    population: list[Candidate],
    tournament_size: int = 3,
) -> Candidate:
    """Select a candidate using tournament selection.

    Picks tournament_size random candidates and returns the one
    with the highest composite_score.
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    k = min(tournament_size, len(population))
    tournament = random.sample(population, k)
    return max(tournament, key=lambda c: c.composite_score)


def crossover(
    parent_a: Candidate,
    parent_b: Candidate,
    task_description: str,
    llm_client=None,
) -> Candidate:
    """Create a child candidate by combining two parents.

    If llm_client is provided, uses LLM to intelligently combine prompts.
    Otherwise, uses simple swap crossover.
    """
    if llm_client:
        prompt = CROSSOVER_PROMPT.format(
            task_description=task_description,
            sys_a=parent_a.prompts.system_prompt,
            user_a=parent_a.prompts.user_prompt,
            sys_b=parent_b.prompts.system_prompt,
            user_b=parent_b.prompts.user_prompt,
        )
        try:
            response = llm_client.chat(
                system_prompt="You are a prompt engineering expert. Return only valid JSON.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=2048,
            )
            parsed = _parse_json_response(response.content)
            if parsed and isinstance(parsed, dict):
                return Candidate(
                    prompts=PromptPair(
                        system_prompt=parsed.get("system_prompt", parent_a.prompts.system_prompt),
                        user_prompt=parsed.get("user_prompt", parent_b.prompts.user_prompt),
                    ),
                    parent_id=parent_a.id,
                    mutation_type="crossover_llm",
                )
        except Exception:
            pass

    # Simple crossover: swap system prompts
    return Candidate(
        prompts=PromptPair(
            system_prompt=parent_a.prompts.system_prompt,
            user_prompt=parent_b.prompts.user_prompt,
        ),
        parent_id=parent_a.id,
        mutation_type="crossover_swap",
    )


def mutate_prompt(
    llm_client,
    candidate: Candidate,
    task_description: str,
    mutation_rate: float = 0.3,
) -> Candidate:
    """Mutate a candidate's prompts using the LLM."""
    if random.random() > mutation_rate:
        return candidate  # No mutation this round

    prompt = MUTATION_PROMPT.format(
        task_description=task_description,
        system_prompt=candidate.prompts.system_prompt,
        user_prompt=candidate.prompts.user_prompt,
    )

    try:
        response = llm_client.chat(
            system_prompt="You are a prompt engineering expert. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.9,  # Higher temp for more creative mutations
            max_tokens=2048,
        )
        parsed = _parse_json_response(response.content)
        if parsed and isinstance(parsed, dict):
            return Candidate(
                prompts=PromptPair(
                    system_prompt=parsed.get("system_prompt", candidate.prompts.system_prompt),
                    user_prompt=parsed.get("user_prompt", candidate.prompts.user_prompt),
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


def evolve_population(
    llm_client,
    population: list[Candidate],
    task_description: str,
    config: ExperimentConfig,
) -> list[Candidate]:
    """Create the next generation from the current population.

    Steps:
    1. Select elites (top performer survives)
    2. Tournament selection for parents
    3. Crossover to create children
    4. Mutate some children
    """
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
            # Crossover
            parent_a = tournament_select(population, config.tournament_size)
            parent_b = tournament_select(population, config.tournament_size)
            child = crossover(parent_a, parent_b, task_description, llm_client)
        else:
            # Select one parent and mutate
            parent = tournament_select(population, config.tournament_size)
            child = mutate_prompt(llm_client, parent, task_description, config.mutation_rate)

        child.generation = next_gen_number
        next_generation.append(child)

    return next_generation
