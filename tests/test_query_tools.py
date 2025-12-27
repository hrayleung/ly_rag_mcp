"""
Tests for rag.tools.query.
"""

import pytest

from rag.models import RetrievalResult, SearchMode, SearchResult
from rag.tools.query import _search_project, _format_result, RELEVANCE_THRESHOLD


class DummyEngine:
    def __init__(self, result):
        self._result = result

    def search(self, **kwargs):
        return self._result


def _mk_result(scores):
    results = [
        RetrievalResult(text=f"doc-{i}", score=s, metadata={}, node_id=str(i))
        for i, s in enumerate(scores)
    ]
    return SearchResult(
        results=results,
        query="q",
        search_mode=SearchMode.SEMANTIC,
        reranked=False,
        used_hyde=False,
        generated_query=None,
        project="p",
    )


def test_relevance_uses_max_score_across_all_results():
    result = _mk_result([0.1, 0.6, 0.2])
    engine = DummyEngine(result)

    _, is_relevant = _search_project(engine, "question", "proj")

    assert is_relevant is True


def test_relevance_all_none_scores_not_relevant():
    result = _mk_result([None, None])
    engine = DummyEngine(result)

    _, is_relevant = _search_project(engine, "q", "proj")

    assert is_relevant is False


def test_format_result_handles_none_scores_text_output():
    result = _mk_result([None, 0.4])
    output = _format_result(result, project="proj", auto_routed=False, return_metadata=False)

    assert "score: n/a" in output
    assert "Found" in output


def test_format_result_empty_results_returns_error_metadata():
    empty = SearchResult(
        results=[],
        query="q",
        search_mode=SearchMode.SEMANTIC,
        reranked=False,
        used_hyde=False,
        generated_query=None,
        project="proj",
    )

    output = _format_result(empty, project="proj", auto_routed=False, return_metadata=True)

    assert isinstance(output, dict)
    assert "error" in output
    assert output["project"] == "proj"
