import types

import pytest

from rag.retrieval import hyde
from rag.retrieval.hyde import generate_hyde_query


def setup_function(_):
    hyde._hyde_cache.clear()


def test_generate_hyde_uses_configured_timeout(monkeypatch):
    calls = {}

    class FakeLLM:
        def __init__(self, model, api_key, timeout):
            calls["timeout"] = timeout

        def complete(self, prompt):
            calls["prompt"] = prompt
            return "synthetic"

    def fake_openai(model, api_key, timeout):
        return FakeLLM(model, api_key, timeout)

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr("llama_index.llms.openai.OpenAI", fake_openai)

    result = generate_hyde_query("what is hyde?")

    assert result == "synthetic"
    assert calls["timeout"] == hyde.settings.hyde_timeout
    assert "Question: what is hyde?" in calls["prompt"]


def test_generate_hyde_retries_then_succeeds(monkeypatch):
    attempts = {"count": 0}

    class FlakyLLM:
        def __init__(self, model, api_key, timeout):
            self.timeout = timeout

        def complete(self, prompt):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary failure")
            return "recovered"

    def flaky_openai(model, api_key, timeout):
        return FlakyLLM(model, api_key, timeout)

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr("llama_index.llms.openai.OpenAI", flaky_openai)
    monkeypatch.setattr(hyde, "INITIAL_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(hyde.time, "sleep", lambda _seconds: None)

    result = generate_hyde_query("retry please")

    assert result == "recovered"
    assert attempts["count"] == 2


def test_generate_hyde_uses_cache(monkeypatch):
    calls = {"count": 0}

    class CountingLLM:
        def __init__(self, model, api_key, timeout):
            pass

        def complete(self, prompt):
            calls["count"] += 1
            return f"resp-{calls['count']}"

    def counting_openai(model, api_key, timeout):
        return CountingLLM(model, api_key, timeout)

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr("llama_index.llms.openai.OpenAI", counting_openai)

    first = generate_hyde_query("cache me")
    second = generate_hyde_query("cache me")

    assert first == second == "resp-1"
    assert calls["count"] == 1
