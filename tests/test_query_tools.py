"""
Tests for rag.tools.query.
"""

import pytest

from rag.models import RetrievalResult, SearchMode, SearchResult
from rag.tools.query import _search_project, _format_result, RELEVANCE_THRESHOLD_HIGH, sanitize_question


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


def test_explicit_project_learning(monkeypatch):
    """Bug M2: Verify learning happens for explicit project queries regardless of relevance score."""
    from rag.project.metadata import MetadataManager
    import rag.tools.query as query_mod

    # Track calls to learn_from_query
    learn_calls = []

    class FakeMCP:
        def __init__(self):
            self.tools = {}

        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    class FakeMetadataManager:
        def learn_from_query(self, project, question, success=True):
            learn_calls.append((project, question, success))

        def track_chat_turn(self, project):
            pass

    class FakeProjectManager:
        def discover_projects(self):
            return []

    class FakeEngine:
        def search(self, **kwargs):
            # Return results with low scores (below RELEVANCE_THRESHOLD of 0.5)
            return _mk_result([0.1, 0.2, 0.3])

    # Patch the managers
    monkeypatch.setattr(query_mod, "get_metadata_manager", lambda: FakeMetadataManager())
    monkeypatch.setattr(query_mod, "get_project_manager", lambda: FakeProjectManager())
    monkeypatch.setattr(query_mod, "get_search_engine", lambda: FakeEngine())

    # Register query tools
    mcp = FakeMCP()
    query_mod.register_query_tools(mcp)

    # Get the query_rag tool
    query_rag = mcp.tools["query_rag"]

    # Query with explicit project
    result = query_rag(
        question="test question",
        project="my-project",
        return_metadata=True
    )

    # Should have learned even though scores were low (0.3 < 0.5 threshold)
    assert len(learn_calls) == 1
    assert learn_calls[0] == ("my-project", "test question", True)

    # Verify result structure
    assert isinstance(result, dict)
    assert "sources" in result
    assert result["project"] == "my-project"


def test_query_whitespace_trimmed():
    """Bug L5: Verify leading/trailing whitespace is removed from questions."""
    # Test with leading/trailing whitespace
    question, error = sanitize_question("   test question   ")
    assert error is None
    assert question == "test question"

    # Test with only whitespace
    question, error = sanitize_question("   ")
    assert error is not None
    assert "empty" in error.lower()
    assert question is None

    # Test with tabs and newlines
    question, error = sanitize_question("\t\n  test  \n\t")
    assert error is None
    assert question == "test"

    # Test empty string
    question, error = sanitize_question("")
    assert error is not None
    assert question is None

    # Test None
    question, error = sanitize_question(None)
    assert error is not None
    assert question is None


def test_query_too_long():
    """Bug L5: Verify query length is validated."""
    from rag.config import settings

    # Create a query that's too long
    long_query = "a" * (settings.max_query_length + 1)
    question, error = sanitize_question(long_query)

    assert error is not None
    assert "too long" in error.lower()
    assert question is None


def test_query_valid():
    """Bug L5: Verify valid queries pass sanitization."""
    # Normal query
    question, error = sanitize_question("What is the meaning of life?")
    assert error is None
    assert question == "What is the meaning of life?"

    # Query with special characters (should be preserved)
    question, error = sanitize_question("How to use Python's asyncio?")
    assert error is None
    assert "asyncio" in question

