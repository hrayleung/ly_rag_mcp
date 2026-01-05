"""
Tests for API key validation (Bug L6).
"""

import pytest
import os

from rag.config import validate_api_key, require_openai_key, get_cohere_key, get_gemini_key, get_firecrawl_key


def test_validate_api_key_missing():
    """Bug L6: Verify missing API keys are rejected."""
    with pytest.raises(ValueError) as exc_info:
        validate_api_key(None, "TEST_API_KEY")

    assert "not found" in str(exc_info.value)
    assert "TEST_API_KEY" in str(exc_info.value)


def test_validate_api_key_too_short():
    """Bug L6: Verify short API keys are rejected."""
    with pytest.raises(ValueError) as exc_info:
        validate_api_key("short_key", "TEST_API_KEY", min_length=20)

    assert "too short" in str(exc_info.value)
    assert "TEST_API_KEY" in str(exc_info.value)


def test_validate_api_key_valid():
    """Bug L6: Verify valid API keys pass validation."""
    # Valid key (20+ characters)
    valid_key = "sk-" + "a" * 30  # 34 characters
    result = validate_api_key(valid_key, "TEST_API_KEY", min_length=20)

    assert result == valid_key


def test_validate_api_key_whitespace_trimmed():
    """Bug L6: Verify whitespace is trimmed from API keys."""
    key_with_spaces = "  sk-" + "a" * 30 + "  "
    result = validate_api_key(key_with_spaces, "TEST_API_KEY", min_length=20)

    assert result == key_with_spaces.strip()


def test_require_openai_key_missing(monkeypatch):
    """Bug L6: Verify require_openai_key fails when key is missing."""
    # Remove OPENAI_API_KEY from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        require_openai_key()

    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


def test_require_openai_key_valid(monkeypatch):
    """Bug L6: Verify require_openai_key succeeds with valid key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "a" * 40)

    result = require_openai_key()

    assert result.startswith("sk-")
    assert len(result) >= 20


def test_get_cohere_key_missing(monkeypatch):
    """Bug L6: Verify get_cohere_key returns None when missing."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    result = get_cohere_key()

    assert result is None


def test_get_cohere_key_invalid_but_returns(monkeypatch, caplog):
    """Bug L6: Verify get_cohere_key warns but returns key if too short."""
    monkeypatch.setenv("COHERE_API_KEY", "short")

    result = get_cohere_key()

    # Should return the key even if invalid (optional key)
    assert result == "short"
    # Should have logged a warning
    assert any("validation failed" in record.message.lower() for record in caplog.records)


def test_get_cohere_key_valid(monkeypatch):
    """Bug L6: Verify get_cohere_key returns valid key."""
    monkeypatch.setenv("COHERE_API_KEY", "cohere-" + "b" * 30)

    result = get_cohere_key()

    assert result.startswith("cohere-")
    assert len(result) >= 20


def test_get_gemini_key_missing(monkeypatch):
    """Bug L6: Verify get_gemini_key returns None when missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = get_gemini_key()

    assert result is None


def test_get_gemini_key_valid(monkeypatch):
    """Bug L6: Verify get_gemini_key returns valid key."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-" + "c" * 30)

    result = get_gemini_key()

    assert result.startswith("gemini-")
    assert len(result) >= 20


def test_get_firecrawl_key_missing(monkeypatch):
    """Bug L6: Verify get_firecrawl_key returns None when missing."""
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    result = get_firecrawl_key()

    assert result is None


def test_get_firecrawl_key_valid(monkeypatch):
    """Bug L6: Verify get_firecrawl_key returns valid key."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-" + "d" * 30)

    result = get_firecrawl_key()

    assert result.startswith("fc-")
    assert len(result) >= 20
