"""DeepSeek Flash LLM adapter for ab_explorer."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 60.0


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


# Cost per 1M tokens (DeepSeek V4 Flash pricing)
# Input (cache miss): $0.14/1M, Output: $0.28/1M (as of 2026)
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28


def _calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for token usage."""
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 8)


class DeepSeekClient:
    """Client for DeepSeek Flash chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
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
