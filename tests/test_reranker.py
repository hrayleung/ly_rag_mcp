"""Tests for reranker.rerank() method, especially failure handling."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from rag.retrieval.reranker import RerankerManager
from rag.config import settings


@dataclass
class DummyNode:
    score: float | None
    text: str = "test"


def test_rerank_returns_nodes_when_reranker_unavailable():
    """Test that rerank returns nodes unchanged when reranker is None."""
    mgr = RerankerManager()
    nodes = [DummyNode(0.5), DummyNode(0.3)]

    result = mgr.rerank(nodes, query="test", top_n=10)

    assert result == nodes
    assert len(result) == 2


def test_rerank_returns_empty_when_no_nodes():
    """Test that rerank handles empty node list."""
    mgr = RerankerManager()

    result = mgr.rerank([], query="test", top_n=10)

    assert result == []


def test_rerank_failure_caps_results_no_top_n(monkeypatch):
    """Bug L2: Verify rerank failure uses max_top_k as fallback when top_n is None."""
    import rag.retrieval.reranker as reranker_mod

    # Mock Cohere key availability
    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "fake-key")

    mgr = RerankerManager()

    # Create many nodes (more than max_top_k)
    nodes = [DummyNode(score=float(i) / 100, text=f"doc-{i}") for i in range(100, 0, -1)]

    # Mock a reranker that raises an exception
    mock_reranker = MagicMock()
    mock_reranker.postprocess_nodes.side_effect = Exception("Cohere API error")

    with patch.object(mgr, 'get_reranker', return_value=mock_reranker):
        result = mgr.rerank(nodes, query="test query", top_n=None)

    # Should return at most max_top_k (50) nodes, not all 100
    assert len(result) <= settings.max_top_k
    assert len(result) == 50  # max_top_k is 50
    # Should be the first 50 nodes from the original list
    assert result == nodes[:50]


def test_rerank_failure_respects_top_n_when_provided(monkeypatch):
    """Bug L2: Verify rerank failure respects provided top_n on error."""
    import rag.retrieval.reranker as reranker_mod

    # Mock Cohere key availability
    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "fake-key")

    mgr = RerankerManager()

    # Create many nodes
    nodes = [DummyNode(score=float(i) / 100, text=f"doc-{i}") for i in range(100, 0, -1)]

    # Mock a reranker that raises an exception
    mock_reranker = MagicMock()
    mock_reranker.postprocess_nodes.side_effect = Exception("Cohere API error")

    with patch.object(mgr, 'get_reranker', return_value=mock_reranker):
        result = mgr.rerank(nodes, query="test query", top_n=10)

    # Should return exactly top_n nodes
    assert len(result) == 10
    assert result == nodes[:10]


def test_rerank_success_with_top_n(monkeypatch):
    """Test successful reranking with top_n provided."""
    import rag.retrieval.reranker as reranker_mod

    # Mock Cohere key availability
    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "fake-key")

    mgr = RerankerManager()

    nodes = [DummyNode(0.5), DummyNode(0.3), DummyNode(0.1)]

    # Mock successful reranking
    mock_reranker = MagicMock()
    # Return nodes in different order (reranked)
    reranked = [nodes[1], nodes[0], nodes[2]]
    mock_reranker.postprocess_nodes.return_value = reranked

    with patch.object(mgr, 'get_reranker', return_value=mock_reranker):
        result = mgr.rerank(nodes, query="test", top_n=2)

    # Should return top 2 from reranked results
    assert len(result) == 2
    assert result == reranked[:2]


def test_rerank_success_without_top_n(monkeypatch):
    """Test successful reranking without top_n (returns all results)."""
    import rag.retrieval.reranker as reranker_mod

    # Mock Cohere key availability
    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "fake-key")

    mgr = RerankerManager()

    nodes = [DummyNode(0.5), DummyNode(0.3), DummyNode(0.1)]

    # Mock successful reranking
    mock_reranker = MagicMock()
    reranked = [nodes[1], nodes[0], nodes[2]]
    mock_reranker.postprocess_nodes.return_value = reranked

    with patch.object(mgr, 'get_reranker', return_value=mock_reranker):
        result = mgr.rerank(nodes, query="test", top_n=None)

    # Should return all reranked results
    assert result == reranked
