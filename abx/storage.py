"""SQLite storage persistence for ab_explorer experiments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .models import Candidate, Experiment, ExperimentConfig, ExperimentStatus, PromptPair, TestCase, TestSuite


class Storage:
    """SQLite-backed persistence for experiments, candidates, and results."""

    def __init__(self, db_path: str = "ab_explorer.db"):
        self.db_path = db_path
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    task_description TEXT NOT NULL DEFAULT '',
                    config TEXT NOT NULL DEFAULT '{}',
                    test_suite TEXT,
                    status TEXT NOT NULL DEFAULT 'created',
                    current_generation INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    user_prompt TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 0,
                    parent_id TEXT,
                    mutation_type TEXT,
                    scores TEXT NOT NULL DEFAULT '[]',
                    composite_score REAL NOT NULL DEFAULT 0.0,
                    cost REAL NOT NULL DEFAULT 0.0,
                    latency REAL NOT NULL DEFAULT 0.0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    test_case_index INTEGER NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    output TEXT,
                    cost REAL NOT NULL DEFAULT 0.0,
                    latency REAL NOT NULL DEFAULT 0.0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS experiment_winners (
                    experiment_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    PRIMARY KEY (experiment_id, candidate_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_experiment
                    ON candidates(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_candidates_generation
                    ON candidates(experiment_id, generation);
                CREATE INDEX IF NOT EXISTS idx_test_results_candidate
                    ON test_results(candidate_id);
            """)

    def save_experiment(self, experiment: Experiment) -> None:
        """Insert or update an experiment."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (id, name, task_description, config, test_suite, status,
                    current_generation, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment.id,
                    experiment.name,
                    experiment.task_description,
                    experiment.config.model_dump_json(),
                    experiment.test_suite.model_dump_json() if experiment.test_suite else None,
                    experiment.status.value if isinstance(experiment.status, ExperimentStatus) else experiment.status,
                    experiment.current_generation,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load an experiment by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_experiment(row)

    def list_experiments(self) -> list[dict]:
        """List all experiments (summary only)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, name, status, current_generation,
                          created_at, updated_at
                   FROM experiments ORDER BY created_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all related data."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM experiment_winners WHERE experiment_id = ?", (experiment_id,))
            candidate_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM candidates WHERE experiment_id = ?", (experiment_id,)
                ).fetchall()
            ]
            for cid in candidate_ids:
                conn.execute("DELETE FROM test_results WHERE candidate_id = ?", (cid,))
            conn.execute("DELETE FROM candidates WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return conn.total_changes > 0

    def save_candidate(self, candidate: Candidate, experiment_id: str) -> None:
        """Insert or update a candidate."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO candidates
                   (id, experiment_id, system_prompt, user_prompt, generation,
                    parent_id, mutation_type, scores, composite_score, cost,
                    latency, token_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.id,
                    experiment_id,
                    candidate.prompts.system_prompt,
                    candidate.prompts.user_prompt,
                    candidate.generation,
                    candidate.parent_id,
                    candidate.mutation_type,
                    json.dumps(candidate.scores),
                    candidate.composite_score,
                    candidate.cost,
                    candidate.latency,
                    candidate.token_count,
                    candidate.created_at.isoformat(),
                ),
            )

    def get_candidates(self, experiment_id: str, generation: Optional[int] = None) -> list[Candidate]:
        """Load candidates for an experiment, optionally filtered by generation."""
        with self._get_conn() as conn:
            if generation is not None:
                rows = conn.execute(
                    "SELECT * FROM candidates WHERE experiment_id = ? AND generation = ? ORDER BY composite_score DESC",
                    (experiment_id, generation),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM candidates WHERE experiment_id = ? ORDER BY generation DESC, composite_score DESC",
                    (experiment_id,),
                ).fetchall()
            return [self._row_to_candidate(r) for r in rows]

    def save_test_result(self, candidate_id: str, test_case_index: int, score: float,
                         output: str, cost: float, latency: float, token_count: int) -> None:
        """Save a single test result for a candidate."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO test_results
                   (candidate_id, test_case_index, score, output, cost, latency, token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, test_case_index, score, output, cost, latency, token_count),
            )

    def save_winner(self, experiment_id: str, candidate_id: str, rank: int, generation: int) -> None:
        """Record a winning candidate at a given generation."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiment_winners
                   (experiment_id, candidate_id, rank, generation)
                   VALUES (?, ?, ?, ?)""",
                (experiment_id, candidate_id, rank, generation),
            )

    def get_experiment_stats(self, experiment_id: str) -> dict:
        """Get aggregate statistics for an experiment.

        Returns a dict with:
            total_cost (float): Sum of all candidate costs
            total_tokens (int): Sum of all candidate token counts
            total_candidates (int): Number of candidates evaluated
            total_generations (int): Number of generations completed
            avg_cost (float): Average cost per candidate
            avg_latency (float): Average latency per candidate
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT
                       COALESCE(SUM(cost), 0.0) as total_cost,
                       COALESCE(SUM(token_count), 0) as total_tokens,
                       COUNT(*) as total_candidates,
                       COALESCE(AVG(latency), 0.0) as avg_latency
                   FROM candidates
                   WHERE experiment_id = ?""",
                (experiment_id,),
            ).fetchone()

            exp_row = conn.execute(
                "SELECT current_generation FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
            total_gens = exp_row["current_generation"] if exp_row else 0

            total_cost = row["total_cost"]
            total_candidates = row["total_candidates"]

            return {
                "total_cost": total_cost,
                "total_tokens": row["total_tokens"],
                "total_candidates": total_candidates,
                "total_generations": total_gens,
                "avg_cost": total_cost / total_candidates if total_candidates > 0 else 0.0,
                "avg_latency": row["avg_latency"],
            }

    def get_winners(self, experiment_id: str) -> list[Candidate]:
        """Get all winning candidates for an experiment."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT c.* FROM candidates c
                   JOIN experiment_winners w ON c.id = w.candidate_id
                   WHERE w.experiment_id = ?
                   ORDER BY w.generation DESC, w.rank ASC""",
                (experiment_id,),
            ).fetchall()
            return [self._row_to_candidate(r) for r in rows]

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        config = ExperimentConfig(**json.loads(row["config"]))
        test_suite = None
        if row["test_suite"]:
            try:
                ts_data = json.loads(row["test_suite"])
                test_suite = TestSuite(**ts_data)
            except (json.JSONDecodeError, TypeError):
                pass
        return Experiment(
            id=row["id"],
            name=row["name"],
            task_description=row["task_description"],
            config=config,
            test_suite=test_suite,
            status=ExperimentStatus(row["status"]),
            current_generation=row["current_generation"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_candidate(self, row: sqlite3.Row) -> Candidate:
        return Candidate(
            id=row["id"],
            prompts=PromptPair(
                system_prompt=row["system_prompt"],
                user_prompt=row["user_prompt"],
            ),
            generation=row["generation"],
            parent_id=row["parent_id"],
            mutation_type=row["mutation_type"],
            scores=json.loads(row["scores"]),
            composite_score=row["composite_score"],
            cost=row["cost"],
            latency=row["latency"],
            token_count=row["token_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
