"""LLM adapters for ab_explorer.

Two providers, picked with `--provider` on the CLI (see `make_client`):

- ``claude`` (default): shells out to the Claude Code CLI (`claude -p`), passing
  the system prompt via --system-prompt-file and the user prompt on stdin.
  Auth and billing come from the local Claude Code login; no API key.
- ``deepseek``: DeepSeek chat completions over HTTP; needs DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx


DEFAULT_MODEL = "haiku"
DEFAULT_TIMEOUT = 300.0
DEFAULT_CLI = "claude"

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_DEFAULT_TIMEOUT = 60.0

PROVIDERS = ("claude", "deepseek")
DEFAULT_PROVIDER = "claude"


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float


class LLMClient(Protocol):
    """What the rest of abx needs from an LLM client."""

    model: str

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class ClaudeCLIClient:
    """Client that runs prompts through the Claude Code CLI in print mode."""

    def __init__(
        self,
        model: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        cli_path: Optional[str] = None,
    ):
        self.model = model or os.getenv("ABX_CLAUDE_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.cli_path = cli_path or os.getenv("ABX_CLAUDE_CLI") or shutil.which(DEFAULT_CLI)

        if not self.cli_path:
            raise ValueError(
                "claude CLI not found on PATH. Install Claude Code or set ABX_CLAUDE_CLI."
            )

    def _build_command(self, system_prompt_file: Optional[str]) -> list[str]:
        cmd = [
            self.cli_path,
            "-p",
            "--model", self.model,
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--setting-sources", "",
            "--strict-mcp-config",
        ]
        if system_prompt_file:
            cmd += ["--system-prompt-file", system_prompt_file]
        return cmd

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Run one prompt through `claude -p`.

        Args:
            system_prompt: Optional system-level instruction.
            user_prompt: The user message.
            temperature: Accepted for interface compatibility; the CLI has no
                sampling controls, so it is ignored.
            max_tokens: Accepted for interface compatibility; ignored.

        Returns:
            LLMResponse with content, token counts, latency, and cost.
        """
        # Run from an empty dir so no project CLAUDE.md/AGENTS.md leaks into context.
        with tempfile.TemporaryDirectory(prefix="abx-claude-") as workdir:
            sp_file = None
            if system_prompt:
                sp_file = os.path.join(workdir, "system_prompt.txt")
                with open(sp_file, "w", encoding="utf-8") as f:
                    f.write(system_prompt)

            start = time.monotonic()
            proc = subprocess.run(
                self._build_command(sp_file),
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=workdir,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:500]}"
            )
        if proc.returncode != 0 or data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(data.get('result', ''))[:500]}")

        usage = data.get("usage", {})
        prompt_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        completion_tokens = usage.get("output_tokens", 0)
        model_usage = data.get("modelUsage") or {}

        return LLMResponse(
            content=data.get("result") or "",
            model=next(iter(model_usage), self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(data.get("duration_ms", elapsed_ms), 2),
            cost=round(float(data.get("total_cost_usd", 0.0)), 8),
        )


# Cost per 1M tokens (DeepSeek V4 Flash pricing)
# Input (cache miss): $0.14/1M, Output: $0.28/1M (as of 2026)
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28


def _calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate DeepSeek cost in USD for token usage."""
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 8)


class DeepSeekClient:
    """Client for DeepSeek Flash chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        model: str = "",
        timeout: float = DEEPSEEK_DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Provide api_key or set DEEPSEEK_API_KEY env var."
            )

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request to DeepSeek Flash.

        Args:
            system_prompt: Optional system-level instruction.
            user_prompt: The user message.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with content, token counts, latency, and cost.
        """
        messages = self._build_messages(system_prompt, user_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.monotonic()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        elapsed_ms = (time.monotonic() - start) * 1000

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(elapsed_ms, 2),
            cost=_calculate_cost(prompt_tokens, completion_tokens),
        )


def make_client(provider: str = "", model: str = "") -> LLMClient:
    """Build the client for `provider` (falls back to $ABX_PROVIDER, then claude).

    An empty `model` lets each client pick its own default.
    Raises ValueError for an unknown provider or a client that can't start.
    """
    provider = (provider or os.getenv("ABX_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider == "claude":
        return ClaudeCLIClient(model=model)
    if provider == "deepseek":
        return DeepSeekClient(model=model)
    raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
