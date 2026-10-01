---
name: abx-setup
description: Install and verify ab_explorer (the `abx` CLI) — virtualenv, editable install, LLM provider check (Claude CLI or DEEPSEEK_API_KEY), smoke test, and running the pytest suite. Use when asked to set up, install, configure, or check that abx works, or when an abx command fails with "command not found", a missing module, or a missing `claude` CLI / DeepSeek API key.
---

# abx setup

ab_explorer is a Python 3.11+ package (`abx/`) exposing the `abx` Typer CLI. LLM calls go through `abx/llm.py` `make_client`: `--provider claude` (default; local Claude Code CLI, no key) or `--provider deepseek` (`DEEPSEEK_API_KEY`). `$ABX_PROVIDER` sets the default.

## 1. Check the environment

```bash
python3 --version            # needs 3.11+
ls venv/ .venv/ 2>/dev/null  # an existing virtualenv?
claude --version              # claude provider: CLI on PATH and logged in?
echo "${DEEPSEEK_API_KEY:+set}"  # deepseek provider: key set?
```

There is no global `python` on this machine and the system `python3` has no pytest, so always work inside a virtualenv.

## 2. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

`venv/`, `.venv/`, `*.db` and `.env` are already in `.gitignore`.

## 3. Check the provider

abx shells out to `claude -p --model <m> --output-format json --tools "" --system-prompt-file <f>` with the user prompt on stdin, from an empty temp dir. Auth and billing come from the user's Claude Code login. Optional env vars: `ABX_CLAUDE_MODEL` (default `haiku`) and `ABX_CLAUDE_CLI` (binary path). The CLI has no temperature/max_tokens controls, so abx ignores those.

For DeepSeek: `export DEEPSEEK_API_KEY="..."` and pass `--provider deepseek` (or `export ABX_PROVIDER=deepseek`). Never write the key into a tracked file. `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` are honoured when `--model` is empty.

## 4. Verify

```bash
abx --help                      # lists init, run, report, list-experiments, generate-tests, self-optimize, self-eval-compare
pytest -q                       # all LLM calls are mocked; no key or network needed
pytest --cov=abx                # optional coverage
```

Tests passing proves the install, not the CLI login. To prove the CLI works without spending much, run one tiny `generate-tests` call (see the `abx-generate-tests` skill) with `--count 1`, and tell the user it makes a real, billed API call before running it.

## 5. Report back

Tell the user: Python version, where the venv is, whether `claude` is found and logged in, and the pytest result (pass/fail counts, with any failure output).
