"""Tests for abx.population."""

import json
from unittest.mock import MagicMock

import pytest

from abx.models import Candidate, ExperimentConfig, PromptPair
from abx.population import (
    _parse_json_response,
    _strip_code_fence,
    crossover,
    evolve_population,
    generate_initial_population,
    mutate_prompt,
    tournament_select,
)


def make_mock_llm(response_text: str):
    """Create a mock LLM client that returns a fixed response."""
    client = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    client.chat.return_value = resp
    return client


class TestStripCodeFence:
    def test_no_fence(self):
        assert _strip_code_fence("hello") == "hello"

    def test_basic_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _strip_code_fence(text)
        assert result == '{"key": "value"}'

    def test_fence_without_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = _strip_code_fence(text)
        assert result == '{"key": "value"}'


class TestParseJsonResponse:
    def test_direct_json(self):
        result = _parse_json_response('{"system_prompt": "Be helpful"}')
        assert result == {"system_prompt": "Be helpful"}

    def test_json_array(self):
        result = _parse_json_response('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_code_fence_json(self):
        text = '```json\n{"system_prompt": "hello"}\n```'
        result = _parse_json_response(text)
        assert result == {"system_prompt": "hello"}

    def test_invalid_returns_none(self):
        result = _parse_json_response("not json at all")
        assert result is None

    def test_embedded_json(self):
        text = "Here is the JSON: {\"key\": \"value\"} and that's it"
        result = _parse_json_response(text)
        assert result == {"key": "value"}


class TestInitialPopulation:
    def test_generates_candidates(self):
        mock_llm = make_mock_llm(json.dumps([
            {"system_prompt": "Be concise", "user_prompt": "Summarize this"},
            {"system_prompt": "Be detailed", "user_prompt": "Explain thoroughly"},
        ]))
        candidates = generate_initial_population(
            mock_llm, "summarize text", population_size=2
        )
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)
        assert candidates[0].prompts.system_prompt == "Be concise"
        assert candidates[0].generation == 0

    def test_fills_remaining(self):
        mock_llm = make_mock_llm(json.dumps([
            {"system_prompt": "Only one", "user_prompt": "task"},
        ]))
        candidates = generate_initial_population(
            mock_llm, "test", population_size=3
        )
        assert len(candidates) == 3  # Filled with defaults

    def test_fallback_on_bad_response(self):
        mock_llm = make_mock_llm("I cannot generate prompts")
        candidates = generate_initial_population(
            mock_llm, "test", population_size=1
        )
        assert len(candidates) == 1
        assert isinstance(candidates[0], Candidate)


class TestTournamentSelect:
    def test_selects_highest_score(self):
        pop = [
            Candidate(prompts=PromptPair(), composite_score=5.0),
            Candidate(prompts=PromptPair(), composite_score=9.0),
            Candidate(prompts=PromptPair(), composite_score=7.0),
        ]
        # With tournament_size=3, all are candidates
        selected = tournament_select(pop, tournament_size=3)
        assert selected.composite_score == 9.0

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            tournament_select([], tournament_size=3)

    def test_single_element(self):
        c = Candidate(prompts=PromptPair(), composite_score=8.0)
        selected = tournament_select([c], tournament_size=3)
        assert selected.composite_score == 8.0


class TestCrossover:
    def test_llm_crossover(self):
        parent_a = Candidate(
            prompts=PromptPair(system_prompt="sys A", user_prompt="user A"),
        )
        parent_b = Candidate(
            prompts=PromptPair(system_prompt="sys B", user_prompt="user B"),
        )
        mock_llm = make_mock_llm(json.dumps({
            "system_prompt": "sys combined",
            "user_prompt": "user combined",
        }))
        child = crossover(parent_a, parent_b, "test task", llm_client=mock_llm)
        assert child.prompts.system_prompt == "sys combined"
        assert child.prompts.user_prompt == "user combined"
        assert child.mutation_type == "crossover_llm"

    def test_simple_crossover(self):
        parent_a = Candidate(
            prompts=PromptPair(system_prompt="sys A", user_prompt="user A"),
        )
        parent_b = Candidate(
            prompts=PromptPair(system_prompt="sys B", user_prompt="user B"),
        )
        child = crossover(parent_a, parent_b, "test task", llm_client=None)
        assert child.prompts.system_prompt == "sys A"
        assert child.prompts.user_prompt == "user B"
        assert child.mutation_type == "crossover_swap"


class TestMutate:
    def test_mutation_applied(self):
        c = Candidate(
            prompts=PromptPair(system_prompt="original sys", user_prompt="original user"),
        )
        mock_llm = make_mock_llm(json.dumps({
            "system_prompt": "mutated sys",
            "user_prompt": "mutated user",
        }))
        # Force mutation with rate=1.0
        mutated = mutate_prompt(mock_llm, c, "test task", mutation_rate=1.0)
        assert mutated.prompts.system_prompt == "mutated sys"
        assert mutated.parent_id == c.id
        assert mutated.mutation_type == "llm_mutation"

    def test_no_mutation_when_rate_zero(self):
        c = Candidate(
            prompts=PromptPair(system_prompt="original", user_prompt="original"),
        )
        mock_llm = make_mock_llm("whatever")
        # rate=0 means never mutate
        mutated = mutate_prompt(mock_llm, c, "test task", mutation_rate=0.0)
        assert mutated.prompts.system_prompt == "original"


class TestEvolvePopulation:
    def test_evolve_increases_generation(self):
        pop = [
            Candidate(prompts=PromptPair(), composite_score=5.0, generation=0),
            Candidate(prompts=PromptPair(), composite_score=9.0, generation=0),
            Candidate(prompts=PromptPair(), composite_score=7.0, generation=0),
            Candidate(prompts=PromptPair(), composite_score=6.0, generation=0),
            Candidate(prompts=PromptPair(), composite_score=8.0, generation=0),
        ]
        mock_llm = make_mock_llm(json.dumps({
            "system_prompt": "evolved sys",
            "user_prompt": "evolved user",
        }))
        config = ExperimentConfig(population_size=5, crossover_rate=0.0)
        next_gen = evolve_population(mock_llm, pop, "test task", config)

        assert len(next_gen) == config.population_size
        assert all(c.generation == 1 for c in next_gen)
        # Elite survives (preserves the best prompts)
        assert next_gen[0].mutation_type == "elite"
        assert next_gen[0].parent_id == pop[1].id  # Best candidate's ID
