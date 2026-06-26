# ab_explorer — Prompt A/B Testing CLI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ab_explorer** is a CLI tool that uses **genetic algorithms** to automatically optimize LLM prompts through A/B testing. It generates a population of prompt variants, evaluates them against a rubric-based test suite, and evolves the best-performing prompts across generations until convergence.

```
abx init --task "Extract dates from text" --tests tests.json
abx run --experiment-id <id> --cycles 20
abx report --experiment-id <id> --winner-only
```

---

## Features

- **🧬 Genetic Algorithm Optimization** — Evolves prompt populations through tournament selection, LLM-powered crossover, and mutation
- **📊 Composite KPI Scoring** — Multi-factor scoring balancing accuracy, cost, and latency with configurable weights
- **🎯 Test Case Generation** — Automatically generate test suites from existing prompts using LLM, with `abx generate-tests`
- **🎯 Rubric-Based Evaluation** — Each test case includes a scoring rubric; the LLM evaluates outputs against it on a 0–10 scale
- **⏱️ Plateau Detection** — Automatically converges when top scores stagnate (configurable threshold and patience)
- **💾 SQLite Persistence** — Full experiment state, candidates, test results, and winners stored in a portable database
- **🔧 Configurable CLI** — Fine-tune population size, mutation rate, crossover rate, KPI weights, and more
- **📈 Rich Console Reports** — Color-coded generation summaries, winner tables, and experiment overviews

---

## Installation

### Prerequisites

- Python 3.11+
- A [DeepSeek](https://platform.deepseek.com/) API key

### Install

```bash
# Clone the repository
git clone https://github.com/Kos-M/ab_explorer
cd ab_explorer

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set your API key
export DEEPSEEK_API_KEY="your-key-here"
```

Optional: add `DEEPSEEK_BASE_URL` or `DEEPSEEK_MODEL` to customize the API endpoint.

---

## Quick Start

### 1. Create a test suite

Create a JSON file with your task description and test cases:

```json
{
  "task_description": "Extract calendar dates from text",
  "evaluation_model": "deepseek-chat",
  "test_cases": [
    {
      "input": "The meeting is on March 5, 2024 at 3pm.",
      "rubric": "Must extract the full date including year"
    },
    {
      "input": "Tomorrow at noon",
      "rubric": "Must identify relative dates and convert to absolute"
    },
    {
      "input": "We meet every Monday at 10am",
      "rubric": "Must identify recurring pattern, not a single date"
    }
  ]
}
```

### 2. Initialize an experiment

```bash
abx init \
  --task "Extract calendar dates from text" \
  --tests tests.json \
  --name "Date Extraction v1"
```

This creates an experiment and prints its ID:
```
✓ Experiment initialized: a1b2c3d4e5f6
  Name: Date Extraction v1
  Task: Extract calendar dates from text
  Test cases: 3
  Database: ab_explorer.db
```

### 3. Run the optimization

```bash
abx run --experiment-id a1b2c3d4e5f6 --cycles 15
```

The optimizer generates an initial population of 5 prompt candidates, then evolves them across generations:

```
🚀 Starting experiment: Date Extraction v1
Cycles: 15 | Population: 5

Generation 0: Generating initial population...
  Evaluating candidate 1/5...
  Evaluating candidate 2/5...
  Best: 0.7234  Avg: 0.6542  Pop: 5

Generation 1: Evolving...
  Evaluating candidate 1/5...
  Best: 0.8112  Avg: 0.7211  Pop: 5

✓ Converged at generation 9!
```

### 4. View the results

```bash
abx report --experiment-id a1b2c3d4e5f6
```

Full report with generation winners table:

```
╭──────────────────────────────┬──────────────────────────────────╮
│ Metric                       │ Value                            │
├──────────────────────────────┼──────────────────────────────────┤
│ ID                           │ a1b2c3d4e5f6                     │
│ Task                         │ Extract calendar dates...        │
│ Status                       │ converged                        │
│ Generations                  │ 9                                │
│ Test Cases                   │ 3                                │
╰──────────────────────────────┴──────────────────────────────────╯

Generation Winners
┌────────────┬─────────┬─────────────┬──────────┐
│ Generation │ Score   │ Cost        │ Latency  │
├────────────┼─────────┼─────────────┼──────────┤
│ 0          │ 0.7234  │ $0.000237   │ 1842ms   │
│ 1          │ 0.8112  │ $0.000198   │ 1651ms   │
│ ...        │ ...     │ ...         │ ...      │
│ 9          │ 0.9451  │ $0.000156   │ 1210ms   │
└────────────┴─────────┴─────────────┴──────────┘
```

### 5. Generate test cases from existing prompts

If you already have prompts and want to generate a test suite automatically:

```bash
abx generate-tests \
  --task "Extract calendar dates from text" \
  --system-prompt "You are a date extraction specialist." \
  --user-prompt "Extract all dates from: {input}" \
  --count 5 \
  --output tests.json
```

```
✓ Generated 5 test cases
  Output: /path/to/tests.json

  Use with: abx init --task "Extract calendar dates from text" --tests tests.json
```

The LLM analyzes your prompts and generates diverse test cases covering easy, medium, and hard scenarios. The output tests.json can be fed directly into `abx init`.

Or with `--winner-only`:

```bash
abx report --experiment-id a1b2c3d4e5f6 --winner-only
```

```
=== Winning Prompt ===

System Prompt:
You are a date extraction specialist. Given any text, identify all date expressions—
absolute dates (March 5, 2024), relative dates (tomorrow, next week), and recurring
patterns (every Monday). Return each date in ISO format with its context.

User Prompt:
Extract all dates from the following text. For each date, provide:
1. The original text span
2. The normalized ISO date
3. The type (absolute/relative/recurring)

Text: {input}

Score: 0.9451
Cost: $0.000156
Latency: 1210ms
```

---

## CLI Reference

### `abx init`

Initialize a new experiment.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--task` | `-t` | string | *required* | Task description for the experiment |
| `--tests` | `-f` | string | *required* | Path to test suite JSON file |
| `--name` | `-n` | string | auto-generated | Optional experiment name |
| `--output` | `-o` | string | `ab_explorer.db` | SQLite database path |

### `abx run`

Run the optimization loop for an existing experiment.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--experiment-id` | `-e` | string | *required* | Experiment ID to run |
| `--cycles` | `-c` | int | `20` | Maximum optimization cycles |
| `--population` | `-p` | int | `5` | Population size per generation |
| `--model` | `-m` | string | `deepseek-chat` | DeepSeek model name |
| `--db` | `-d` | string | `ab_explorer.db` | SQLite database path |
| `--accuracy-weight` | | float | `0.5` | KPI accuracy weight |
| `--cost-weight` | | float | `0.3` | KPI cost weight |
| `--latency-weight` | | float | `0.2` | KPI latency weight |
| `--plateau-threshold` | | float | `0.02` | Convergence threshold (2%) |
| `--plateau-rounds` | | int | `5` | Rounds before convergence |

### `abx report`

View experiment results.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--experiment-id` | `-e` | string | *required* | Experiment ID |
| `--db` | `-d` | string | `ab_explorer.db` | SQLite database path |
| `--winner-only` | `-w` | flag | `false` | Show only the winning prompt |

### `abx list-experiments`

List all experiments in the database.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--db` | `-d` | string | `ab_explorer.db` | SQLite database path |

### `abx generate-tests`

Generate a test suite JSON file from existing prompts using LLM.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--task` | `-t` | string | *required* | Task description for the experiment |
| `--system-prompt` | `-s` | string | *required* | System prompt (inline text or path to a `.txt` file) |
| `--user-prompt` | `-u` | string | *required* | User prompt (inline text or path to a `.txt` file) |
| `--output` | `-o` | string | `tests.json` | Output path for the generated test suite |
| `--count` | `-c` | int | `5` | Number of test cases to generate (1–20) |
| `--model` | `-m` | string | `deepseek-v4-flash` | DeepSeek model for test generation |

---

## Test Suite JSON Format

The test suite file defines the task and evaluation rubric:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_description` | string | yes | High-level description of the task |
| `evaluation_model` | string | no | Model used for evaluation (default: `deepseek-chat`) |
| `test_cases` | array | yes | Array of test case objects (min 1) |

Each test case:

| Field | Type | Description |
|-------|------|-------------|
| `input` | string | The input to feed the candidate prompt |
| `rubric` | string | Scoring criteria (0–10) for evaluating the output |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (Typer)                             │
│            abx init / run / report / generate-tests             │
└──────┬──────────────────────────┬───────────────────────────────┘
       │                          │
       ▼                          ▼
┌──────────────┐         ┌──────────────┐
│ Experiment   │         │ Experiment   │
│ Runner       │ ◄─────► │ Config       │
│ (GA Loop)    │         │ (Settings)   │
└──────┬───────┘         └──────────────┘
       │
       ├────────────────────┬──────────────────────┐
       ▼                    ▼                      ▼
┌──────────────┐    ┌──────────────┐     ┌──────────────────┐
│ Population   │    │  Evaluator   │     │  KPI Scorer      │
│ (GA Ops)     │    │ (Rubric-     │     │ (Composite       │
│              │    │  based LLM)  │     │  Weighted Score) │
└──────┬───────┘    └──────┬───────┘     └────────┬─────────┘
       │                   │                       │
       └───────────────────┼───────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  LLM Client  │
                    │  (DeepSeek   │
                    │   Flash)     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Storage    │
                    │  (SQLite)    │
                    └──────────────┘
```

### Module Overview

| Module | File | Role |
|--------|------|------|
| `cli.py` | CLI entry point | Typer commands: `init`, `run`, `report`, `list-experiments`, `generate-tests` |
| `models.py` | Data models | Pydantic v2 schemas: `Experiment`, `Candidate`, `TestSuite`, `PromptPair`, `ExperimentConfig` |
| `experiment.py` | GA loop | `ExperimentRunner` — orchestrates the optimization lifecycle |
| `population.py` | GA operations | Initial generation, tournament selection, crossover, mutation, evolution |
| `evaluator.py` | Rubric scoring | Evaluates each candidate's output against test case rubrics via LLM |
| `kpi.py` | Composite scoring | Computes weighted KPI: accuracy * cost * latency |
| `llm.py` | LLM adapter | DeepSeek Flash client with `httpx`, token tracking, and cost calculation |
| `storage.py` | Persistence | SQLite CRUD for experiments, candidates, test results, and winners |
| `test_generator.py` | Test generation | Generates `TestSuite` from existing prompts using LLM via `generate_test_suite()` |
| `utils.py` | Utilities | Helper functions including `resolve_system_prompt()` for file-or-inline prompt resolution |

### Genetic Algorithm Flow

```
                     ┌──────────────┐
                     │   Start      │
                     └──────┬───────┘
                            ▼
              ┌─────────────────────────┐
              │ Generate Initial Pop    │  LLM creates N diverse prompt strategies
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
              │ Evaluate Each Candidate │  Run each prompt on all test cases
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
              │ Compute KPI Scores      │  Weighted accuracy x cost x latency
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
         ┌───►│ Selection (Tournament)  │  Pick best from random subsets
         │    └──────────┬──────────────┘
         │               ▼
         │    ┌─────────────────────────┐
         │    │ Crossover / Mutation    │  LLM-powered recombination + tweaks
         │    └──────────┬──────────────┘
         │               ▼
         │    ┌─────────────────────────┐
         │    │ New Generation          │  Elitism: best candidate survives
         │    └──────────┬──────────────┘
         │               ▼
         │    ┌─────────────────────────┐
         │    │ Evaluate + Score        │
         │    └──────────┬──────────────┘
         │               ▼
         │    ┌─────────────────────────┐
         │    │ Check Convergence       │  Plateau detection for early stop
         │    └──────────┬──────────────┘
         │               │
         └────── No ─────┤
                         │ Yes
                         ▼
              ┌─────────────────────────┐
              │   Done / Report         │
              └─────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | *Required.* DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Custom API base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name override |

### ExperimentConfig (Default)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `deepseek-chat` | LLM model for generation |
| `cycles` | `20` | Maximum generations |
| `population_size` | `5` | Candidates per generation |
| `tournament_size` | `3` | Candidates in each tournament |
| `mutation_rate` | `0.3` | Probability of mutation |
| `crossover_rate` | `0.5` | Probability of crossover vs mutation |
| `plateau_threshold` | `0.02` | Max score variance for convergence |
| `plateau_rounds` | `5` | Consecutive rounds below threshold |
| `max_tokens` | `4096` | Max tokens per LLM response |
| `kpi_weights` | `{"accuracy": 0.5, "cost": 0.3, "latency": 0.2}` | Composite KPI weights |

### KPI Formula

```
composite = w_a x (avg_score / 10)
          + w_c x (1 - relative_cost)
          + w_l x (1 - relative_latency)

where:
  relative_cost    = cost / max(all_costs)
  relative_latency = latency / max(all_latencies)
```

---

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=abx

# Verbose output
pytest -v
```

### Project Structure

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
│   ├── storage.py        # SQLite persistence
│   ├── test_generator.py # Test case generation from prompts
│   └── utils.py          # Utility functions
├── tests/                # Test suite
│   ├── test_cli.py
│   ├── test_evaluator.py
│   ├── test_experiment.py
│   ├── test_kpi.py
│   ├── test_llm.py
│   ├── test_models.py
│   ├── test_population.py
│   ├── test_storage.py
│   ├── test_test_generator.py
│   └── test_utils.py
├── AGENTS.md             # Agent context metadata
├── README.md             # This file
├── pyproject.toml        # Project config + dependencies
└── .gitignore
```

---

## License

MIT
