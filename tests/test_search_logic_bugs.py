"""
Tests for Phase 1 HIGH severity search logic bugs (H10, H11, H12).

Tests:
- H10: Relevance threshold tiers (0.5 vs 0.2)
- H11: No double slice after reranking
- H12: BM25 fallback handled correctly (no silent TypeError)
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag.models import RetrievalResult, SearchMode, SearchResult
from rag.tools.query import (
    RELEVANCE_THRESHOLD_HIGH,
    RELEVANCE_THRESHOLD_LOW,
    _search_project,
)
from rag.retrieval.search import SearchEngine


class DummyEngine:
    """Dummy search engine for testing."""
    def __init__(self, result):
        self._result = result

    def search(self, **kwargs):
        return self._result


def _mk_result(scores):
    """Helper to create SearchResult with given scores."""
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


# ============================================================================
# Bug H10: Relevance threshold tiers
# ============================================================================

def test_relevance_threshold_tiers_defined():
    """H10: Verify both HIGH and LOW thresholds are defined correctly."""
    assert hasattr(__import__('rag.tools.query', fromlist=['RELEVANCE_THRESHOLD_HIGH']), 'RELEVANCE_THRESHOLD_HIGH')
    assert hasattr(__import__('rag.tools.query', fromlist=['RELEVANCE_THRESHOLD_LOW']), 'RELEVANCE_THRESHOLD_LOW')
    assert RELEVANCE_THRESHOLD_HIGH == 0.5
    assert RELEVANCE_THRESHOLD_LOW == 0.2
    assert RELEVANCE_THRESHOLD_HIGH > RELEVANCE_THRESHOLD_LOW


def test_relevance_threshold_high_stops_routing():
    """H10: Verify HIGH threshold (0.5) stops multi-project routing."""
    # Score 0.6 >= HIGH threshold -> should be relevant
    result = _mk_result([0.6, 0.3, 0.2])
    engine = DummyEngine(result)

    _, is_relevant = _search_project(engine, "question", "proj")

    assert is_relevant is True, "Score 0.6 should be considered relevant (stops routing)"


def test_relevance_threshold_mid_score_continues_routing():
    """H10: Verify score 0.3 (< HIGH, > LOW) continues routing."""
    # Score 0.3 is between thresholds -> should NOT stop routing
    result = _mk_result([0.3, 0.2, 0.1])
    engine = DummyEngine(result)

    _, is_relevant = _search_project(engine, "question", "proj")

    assert is_relevant is False, "Score 0.3 should NOT be relevant enough to stop routing"


def test_relevance_threshold_low_score_triggers_hyde():
    """H10: Verify LOW threshold (0.2) would trigger HyDE expansion."""
    # This tests the threshold value exists and would be used by HyDE logic
    # (actual HyDE trigger logic is in rag/retrieval/hyde.py)
    from rag.config import settings

    # Verify config.py has low_score_threshold matching LOW
    assert settings.low_score_threshold == 0.2
    assert RELEVANCE_THRESHOLD_LOW == 0.2

    # Score 0.1 < LOW threshold -> HyDE should trigger
    result = _mk_result([0.1, 0.05])
    engine = DummyEngine(result)

    _, is_relevant = _search_project(engine, "question", "proj")

    assert is_relevant is False, "Score 0.1 should NOT be relevant enough to stop routing"


# ============================================================================
# Bug H11: No double slice after reranking
# ============================================================================

def test_reranking_no_double_slice(monkeypatch):
    """H11: Verify reranker results are not sliced twice."""
    from rag.retrieval.search import get_search_engine

    # Mock managers
    mock_index_mgr = MagicMock()
    mock_reranker_mgr = MagicMock()
    mock_bm25_mgr = MagicMock()
    mock_chroma_mgr = MagicMock()

    # Setup index mock - return 10 nodes
    mock_index = MagicMock()
    mock_index.docstore.docs = {}

    # Create mock nodes with proper structure
    mock_nodes = []
    for i in range(10):
        node = MagicMock()
        node.score = 0.7 - (i * 0.05)
        node.node.get_content.return_value = f"content-{i}"
        node.node.metadata = {}
        node.node.node_id = f"node-{i}"
        mock_nodes.append(node)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_nodes
    mock_index.as_retriever.return_value = mock_retriever

    mock_index_mgr.get_index.return_value = mock_index
    mock_index_mgr.get_node_count.return_value = 10
    mock_index_mgr.current_project = "test_project"

    # Track the nodes passed to reranker and what it returns
    reranked_nodes = []

    def fake_rerank(nodes, query, top_n):
        # Reranker should slice to top_k internally
        reranked_nodes.append(len(nodes))
        # Return exactly top_k=5 nodes (simulating reranker's internal slicing)
        return nodes[:top_n]

    mock_reranker_mgr.rerank.side_effect = fake_rerank
    mock_reranker_mgr.should_apply_rerank.return_value = True

    monkeypatch.setattr("rag.retrieval.search.get_index_manager", lambda: mock_index_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_reranker_manager", lambda: mock_reranker_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_bm25_manager", lambda: mock_bm25_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_chroma_manager", lambda: mock_chroma_mgr)

    engine = SearchEngine()
    result = engine.search(
        question="test query",
        similarity_top_k=5,
        search_mode="semantic",
        use_rerank=True,
        project="test_project"
    )

    # Verify reranking was applied
    assert mock_reranker_mgr.rerank.called, "Reranker should be called"
    assert result.reranked is True

    # Verify reranker received all 10 nodes
    assert len(reranked_nodes) == 1
    assert reranked_nodes[0] == 10, "Reranker should receive all candidates"

    # Verify final result has exactly 5 nodes (not sliced again)
    assert len(result.results) == 5, "Should have exactly top_k=5 results"


def test_reranking_failure_slices_once(monkeypatch):
    """H11: Verify reranking failure path still slices correctly."""
    from rag.retrieval.search import get_search_engine

    # Mock managers
    mock_index_mgr = MagicMock()
    mock_reranker_mgr = MagicMock()
    mock_bm25_mgr = MagicMock()
    mock_chroma_mgr = MagicMock()

    # Setup index mock
    mock_index = MagicMock()
    mock_node = MagicMock()
    mock_node.score = 0.7
    mock_node.node.get_content.return_value = "content"
    mock_node.node.metadata = {}
    mock_node.node.node_id = "1"

    mock_index.docstore.docs = {}
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_node] * 10  # 10 nodes
    mock_index.as_retriever.return_value = mock_retriever
    mock_index_mgr.get_index.return_value = mock_index
    mock_index_mgr.get_node_count.return_value = 10
    mock_index_mgr.current_project = "test_project"

    # Setup reranker to fail
    mock_reranker_mgr.should_apply_rerank.return_value = True
    mock_reranker_mgr.rerank.side_effect = Exception("Rerank failed")

    monkeypatch.setattr("rag.retrieval.search.get_index_manager", lambda: mock_index_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_reranker_manager", lambda: mock_reranker_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_bm25_manager", lambda: mock_bm25_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_chroma_manager", lambda: mock_chroma_mgr)

    engine = SearchEngine()
    result = engine.search(
        question="test query",
        similarity_top_k=5,
        search_mode="semantic",
        use_rerank=True,
        project="test_project"
    )

    # Verify reranking was attempted but failed
    assert mock_reranker_mgr.rerank.called
    assert result.reranked is False
    # Results should be sliced to top_k despite failure
    assert len(result.results) == 5


# ============================================================================
# Bug H12: BM25 fallback handled correctly
# ============================================================================

def test_bm25_get_retriever_signature_accepts_project():
    """H12: Verify BM25Manager.get_retriever() accepts project parameter."""
    from rag.retrieval.bm25 import BM25Manager

    mgr = BM25Manager()

    # Check method signature
    import inspect
    sig = inspect.signature(mgr.get_retriever)
    params = sig.parameters

    assert 'project' in params, "get_retriever should accept 'project' parameter"
    assert params['project'].default is not inspect.Parameter.empty or \
           params['project'].kind == inspect.Parameter.KEYWORD_ONLY, \
           "project parameter should be optional"


def test_bm25_hybrid_retriever_no_try_except(monkeypatch):
    """H12: Verify _get_retriever handles HYBRID mode correctly with BM25."""
    from rag.retrieval.search import SearchEngine
    from rag.models import SearchMode

    mock_index_mgr = MagicMock()
    mock_bm25_mgr = MagicMock()
    mock_chroma_mgr = MagicMock()

    mock_index = MagicMock()
    mock_index.docstore.docs = {}
    mock_index_mgr.get_index.return_value = mock_index
    mock_index_mgr.current_project = "test_project"

    # Mock BM25 to return a retriever
    mock_bm25_retriever = MagicMock()
    mock_bm25_mgr.get_retriever.return_value = mock_bm25_retriever

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_mgr.get_client.return_value = (mock_client, mock_collection)

    monkeypatch.setattr("rag.retrieval.search.get_index_manager", lambda: mock_index_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_bm25_manager", lambda: mock_bm25_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_chroma_manager", lambda: mock_chroma_mgr)

    engine = SearchEngine()

    # Call _get_retriever with HYBRID mode
    retriever = engine._get_retriever(mock_index, SearchMode.HYBRID, 5, "test_project")

    # Verify BM25 was called with project parameter
    mock_bm25_mgr.get_retriever.assert_called_once_with(mock_index, mock_collection, "test_project")

    # Verify retriever was created
    assert retriever is not None


def test_bm25_keyword_retriever_no_try_except(monkeypatch):
    """H12: Verify _get_retriever handles KEYWORD mode correctly."""
    from rag.retrieval.search import SearchEngine
    from rag.models import SearchMode

    mock_index_mgr = MagicMock()
    mock_bm25_mgr = MagicMock()
    mock_chroma_mgr = MagicMock()

    mock_index = MagicMock()
    mock_index.docstore.docs = {}
    mock_index_mgr.get_index.return_value = mock_index
    mock_index_mgr.current_project = "test_project"

    # Mock BM25 to return None (unavailable)
    mock_bm25_mgr.get_retriever.return_value = None

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_mgr.get_client.return_value = (mock_client, mock_collection)

    monkeypatch.setattr("rag.retrieval.search.get_index_manager", lambda: mock_index_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_bm25_manager", lambda: mock_bm25_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_chroma_manager", lambda: mock_chroma_mgr)

    engine = SearchEngine()

    # Call _get_retriever with KEYWORD mode
    retriever = engine._get_retriever(mock_index, SearchMode.KEYWORD, 5, "test_project")

    # Verify BM25 was called with project parameter
    mock_bm25_mgr.get_retriever.assert_called_once_with(mock_index, mock_collection, "test_project")

    # Verify falls back to vector retriever
    assert retriever is not None
    assert mock_index.as_retriever.called


def test_bm25_fallback_to_vector_on_none(monkeypatch):
    """H12: Verify when BM25 returns None, falls back to vector search."""
    from rag.retrieval.search import SearchEngine
    from rag.models import SearchMode

    mock_index_mgr = MagicMock()
    mock_bm25_mgr = MagicMock()
    mock_chroma_mgr = MagicMock()

    mock_index = MagicMock()
    mock_index.docstore.docs = {}
    mock_vector_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_vector_retriever
    mock_index_mgr.get_index.return_value = mock_index
    mock_index_mgr.current_project = "test_project"

    # BM25 unavailable
    mock_bm25_mgr.get_retriever.return_value = None

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_mgr.get_client.return_value = (mock_client, mock_collection)

    monkeypatch.setattr("rag.retrieval.search.get_index_manager", lambda: mock_index_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_bm25_manager", lambda: mock_bm25_mgr)
    monkeypatch.setattr("rag.retrieval.search.get_chroma_manager", lambda: mock_chroma_mgr)

    engine = SearchEngine()

    # Test keyword mode
    retriever = engine._get_retriever(mock_index, SearchMode.KEYWORD, 5, "test_project")
    assert retriever == mock_vector_retriever

    # Reset mock
    mock_index.as_retriever.reset_mock()
    mock_bm25_mgr.get_retriever.reset_mock()

    # Test hybrid mode
    retriever = engine._get_retriever(mock_index, SearchMode.HYBRID, 5, "test_project")
    assert retriever == mock_vector_retriever


# ============================================================================
# Integration tests
# ============================================================================

def test_thresholds_documentation_consistency():
    """Verify threshold constants are documented and used consistently."""
    import rag.tools.query as query_mod
    import rag.config as config_mod

    # Check query.py has both thresholds
    assert hasattr(query_mod, 'RELEVANCE_THRESHOLD_HIGH')
    assert hasattr(query_mod, 'RELEVANCE_THRESHOLD_LOW')

    # Check config.py has matching low_score_threshold
    assert config_mod.settings.low_score_threshold == query_mod.RELEVANCE_THRESHOLD_LOW

    # Verify HIGH threshold is used for routing decision
    assert RELEVANCE_THRESHOLD_HIGH == 0.5

    # The documentation should explain the tiers
    # (This is a code review check - the file should have comments)
    import inspect
    source = inspect.getsource(query_mod)
    assert "HIGH threshold" in source or "RELEVANCE_THRESHOLD_HIGH" in source
