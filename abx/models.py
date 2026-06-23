"""Pydantic data models for ab_explorer."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PromptType(str, Enum):
    """Whether a prompt is a system prompt or user prompt."""
    SYSTEM = "system"
    USER = "user"


class PromptPair(BaseModel):
    """A pair of system + user prompts."""
    system_prompt: str = ""
    user_prompt: str = ""


class TestCase(BaseModel):
    """A single test case with input and rubric for evaluation."""
    input: str
    rubric: str


class TestSuite(BaseModel):
    """A test suite containing multiple test cases for an experiment."""
    task_description: str
    test_cases: list[TestCase]
    evaluation_model: str = "deepseek-chat"


class Candidate(BaseModel):
    """A prompt candidate in the population, with scoring metadata."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompts: PromptPair
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    scores: list[float] = Field(default_factory=list)
    composite_score: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    token_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

    def avg_score(self) -> float:
        """Average rubric score across all test cases."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


class ExperimentConfig(BaseModel):
    """Configuration for an optimization experiment."""
    model: str = "deepseek-chat"
    cycles: int = 20
    population_size: int = 5
    tournament_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    plateau_threshold: float = 0.02
    plateau_rounds: int = 5
    max_tokens: int = 4096
    kpi_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy": 0.5,
            "cost": 0.3,
            "latency": 0.2,
        }
    )


class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CONVERGED = "converged"


class Experiment(BaseModel):
    """An optimization experiment tracking state across generations."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    task_description: str = ""
    test_suite: Optional[TestSuite] = None
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    status: ExperimentStatus = ExperimentStatus.CREATED
    current_generation: int = 0
    winners: list[Candidate] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
