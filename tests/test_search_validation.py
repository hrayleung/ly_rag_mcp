"""Tests for search input validation and mode selection.

Focuses on SearchEngine._validate_query, _validate_top_k, and _select_search_mode.
"""

from __future__ import annotations

from unittest.mock import ANY, MagicMock

import pytest

from rag.models import SearchMode


@pytest.fixture()
def search_engine(monkeypatch):
    """Create a SearchEngine with dependency managers stubbed."""
    import rag.retrieval.search as search_mod

    monkeypatch.setattr(search_mod, "get_index_manager", lambda: MagicMock())
    monkeypatch.setattr(search_mod, "get_reranker_manager", lambda: MagicMock())
    monkeypatch.setattr(search_mod, "get_bm25_manager", lambda: MagicMock())

    return search_mod.SearchEngine()


def test_validate_query_rejects_non_string(search_engine):
    with pytest.raises(ValueError):
        search_engine._validate_query(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        search_engine._validate_query(123)  # type: ignore[arg-type]


def test_validate_query_rejects_empty_or_whitespace(search_engine):
    with pytest.raises(ValueError):
        search_engine._validate_query("")

    with pytest.raises(ValueError):
        search_engine._validate_query("   \n\t ")


def test_validate_query_strips_and_truncates(search_engine, monkeypatch):
    import rag.retrieval.search as search_mod

    monkeypatch.setattr(search_mod.settings, "max_query_length", 5)

    assert search_engine._validate_query("  hello ") == "hello"
    assert search_engine._validate_query("  hello world  ") == "hello"  # truncated


def test_validate_top_k_defaults_and_coerces(search_engine, monkeypatch):
    import rag.retrieval.search as search_mod

    monkeypatch.setattr(search_mod.settings, "default_top_k", 6)
    monkeypatch.setattr(search_mod.settings, "min_top_k", 1)
    monkeypatch.setattr(search_mod.settings, "max_top_k", 50)

    assert search_engine._validate_top_k(None) == 6
    assert search_engine._validate_top_k("7") == 7  # type: ignore[arg-type]


def test_validate_top_k_clamps_to_min_and_max(search_engine, monkeypatch):
    import rag.retrieval.search as search_mod

    monkeypatch.setattr(search_mod.settings, "default_top_k", 6)
    monkeypatch.setattr(search_mod.settings, "min_top_k", 2)
    monkeypatch.setattr(search_mod.settings, "max_top_k", 4)

    assert search_engine._validate_top_k(1) == 2
    assert search_engine._validate_top_k(999) == 4


def test_select_search_mode_respects_explicit_hybrid_and_keyword(search_engine):
    assert (
        search_engine._select_search_mode("anything", SearchMode.HYBRID)
        == SearchMode.HYBRID
    )
    assert (
        search_engine._select_search_mode("anything", SearchMode.KEYWORD)
        == SearchMode.KEYWORD
    )


@pytest.mark.parametrize(
    "question",
    [
        "HTTP 500 when calling API",  # 3+ digit token
        "How to use FOO_BAR constant",  # [A-Z_]{2,}
        "Why does a.b.c fail?",  # dot token
        "C++ example: if (x) { return y; }",  # code chars
    ],
)
def test_select_search_mode_auto_switches_to_hybrid_for_technical_queries(
    search_engine, question
):
    assert (
        search_engine._select_search_mode(question, SearchMode.SEMANTIC)
        == SearchMode.HYBRID
    )


def test_select_search_mode_uses_uppercase_ratio_threshold(search_engine):
    # 2/3 tokens uppercase -> 0.666 > 0.3
    question = "API TOKEN refresh"
    assert (
        search_engine._select_search_mode(question, SearchMode.SEMANTIC)
        == SearchMode.HYBRID
    )


def test_select_search_mode_uses_digit_ratio_threshold(search_engine):
    # 2/3 tokens contain digits -> 0.666 > 0.4
    question = "error 123 456"
    assert (
        search_engine._select_search_mode(question, SearchMode.SEMANTIC)
        == SearchMode.HYBRID
    )


def test_select_search_mode_defaults_to_semantic(search_engine):
    question = "Explain how authentication works"
    assert (
        search_engine._select_search_mode(question, SearchMode.SEMANTIC)
        == SearchMode.SEMANTIC
    )


def test_bm25_retriever_called_with_collection(monkeypatch):
    import rag.retrieval.search as search_mod

    calls = []

    monkeypatch.setattr(search_mod, "get_index_manager", lambda: MagicMock())
    monkeypatch.setattr(search_mod, "get_reranker_manager", lambda: MagicMock())
    monkeypatch.setattr(search_mod, "get_bm25_manager", lambda: MagicMock())

    class FakeBM25:
        def get_retriever(self, index, collection=None):
            calls.append((index, collection))
            return None

    class FakeChroma:
        def get_client(self, project):
            return None, "collection"

    class FakeIndex:
        def as_retriever(self, similarity_top_k):
            return f"vector-{similarity_top_k}"

    engine = search_mod.SearchEngine()
    engine._bm25_manager = FakeBM25()
    monkeypatch.setattr(search_mod, "get_chroma_manager", lambda: FakeChroma())

    retriever = engine._get_keyword_retriever(FakeIndex(), top_k=5, project="proj")

    assert calls == [(ANY, "collection")]
    assert retriever == "vector-5"
