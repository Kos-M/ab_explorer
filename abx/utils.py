"""Utility functions for ab_explorer."""

from __future__ import annotations

from pathlib import Path


def resolve_system_prompt(value: str) -> str:
    """Resolve a system prompt value — from file path or inline text.

    Checks if *value* is an existing file path and reads its content if so.
    Catches OSError (e.g. [Errno 36] File name too long on Linux when the
    string exceeds PATH_MAX / NAME_MAX) and falls back to treating the
    value as inline text.

    Args:
        value: Either a filesystem path to a prompt file, or inline prompt text.

    Returns:
        The resolved prompt text (from file content or the original value).
    """
    try:
        path = Path(value)
        if path.exists() and path.is_file():
            return path.read_text().strip()
    except OSError:
        pass  # Not a valid path (too long, etc.) — treat as inline text

    return value
