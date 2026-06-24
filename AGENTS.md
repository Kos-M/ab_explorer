# ab_explorer — Prompt A/B Testing CLI Tool

## Directory Structure
```
ab_explorer/
├── abx/                  # Main package
│   ├── __init__.py       # Package marker
│   ├── cli.py            # Typer CLI entry point
│   ├── evaluator.py      # Rubric-based LLM evaluation
│   ├── experiment.py     # Core GA optimization loop
│   ├── kpi.py            # Composite KPI scoring
│   ├── llm.py            # DeepSeek Flash adapter
│   ├── models.py         # Pydantic data models
│   ├── population.py     # Population generation + GA mutation
│   └── storage.py        # SQLite persistence
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_cli.py       # CLI tests (report stats tests)
│   ├── test_evaluator.py
│   ├── test_experiment.py
│   ├── test_kpi.py
│   ├── test_llm.py
│   ├── test_models.py
│   ├── test_population.py
│   └── test_storage.py   # Storage tests (includes stats aggregation)
├── AGENTS.md             # This file
├── pyproject.toml        # Project config + dependencies
└── .gitignore
```

## Tech Stack
- **Language**: Python 3.11+
- **CLI**: Typer + Rich
- **HTTP**: httpx
- **Models**: Pydantic v2
- **Storage**: SQLite (stdlib)
- **Testing**: pytest

## Commands
```bash
# Install
pip install -e ".[dev]"

# Run CLI
abx generate-tests --system-prompt "..." --user-prompt "..." --task "..." --output tests.json --count 5
abx init --task "..." --tests tests.json
abx run --experiment-id <id> --cycles 20
abx report --experiment-id <id> --winner-only
abx list-experiments

# Test
pytest
pytest --cov=abx
```

## Design Decisions
- **Single LLM**: DeepSeek Flash only (no abstract adapter yet)
- **KPI**: Composite with configurable weights (accuracy:cost:latency)
- **Evaluation**: Rubric-based (LLM scores output against rubric on 0-10)
- **Population**: Fully synthetic (LLM generates candidates from task description)
- **Mutation**: Genetic Algorithm (crossover + temperature/instruction tweaks)
- **Convergence**: Plateau detection — stop when top-3 scores vary < 2% for 5 rounds
- **Test Generation**: `generate-tests` command creates tests.json from existing prompts using LLM analysis. Supports inline prompts or file paths. Generates diverse test cases (input + rubric) across difficulty levels.
