"""Tests for abx.utils."""

import os
import tempfile

import pytest

from abx.utils import resolve_system_prompt


class TestResolveSystemPrompt:
    def test_returns_inline_text_when_not_a_file(self):
        """Short inline text should be returned as-is."""
        result = resolve_system_prompt("Be helpful and concise")
        assert result == "Be helpful and concise"

    def test_reads_from_file_when_path_exists(self):
        """When the value points to an existing file, read its content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("You are a math tutor.")
            f.flush()
            tmp_path = f.name

        try:
            result = resolve_system_prompt(tmp_path)
            assert result == "You are a math tutor."
        finally:
            os.unlink(tmp_path)

    def test_strips_whitespace_from_file_content(self):
        """File content should be stripped of leading/trailing whitespace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  \nYou are a coding assistant.\n  ")
            f.flush()
            tmp_path = f.name

        try:
            result = resolve_system_prompt(tmp_path)
            assert result == "You are a coding assistant."
        finally:
            os.unlink(tmp_path)

    def test_handles_long_inline_text_without_oserror(self):
        """Very long inline text should NOT raise OSError (File name too long)."""
        long_text = "Be helpful. " * 200  # ~3000 chars — exceeds NAME_MAX on most FS
        result = resolve_system_prompt(long_text)
        assert result == long_text

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        assert resolve_system_prompt("") == ""

    def test_handles_nonexistent_path_with_short_name(self):
        """A short string that looks like a path but doesn't exist returns as-is."""
        result = resolve_system_prompt("/nonexistent/prompt.txt")
        assert result == "/nonexistent/prompt.txt"

    def test_handles_special_characters_inline(self):
        """Inline text with special characters should pass through unchanged."""
        text = "Role: assistant\nGuidelines: be polite, accurate, & helpful."
        result = resolve_system_prompt(text)
        assert result == text

    def test_handles_multiline_file(self):
        """File content with multiple lines should be read fully."""
        content = "Line 1\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            result = resolve_system_prompt(tmp_path)
            assert result == content
        finally:
            os.unlink(tmp_path)
