"""Tests for reranker decision logic.

Focuses on RerankerManager.should_apply_rerank with and without Cohere key
and score delta threshold behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.retrieval.reranker import RerankerManager


@dataclass
class DummyNode:
    score: float | None


def test_should_apply_rerank_false_when_not_requested():
    mgr = RerankerManager()
    assert mgr.should_apply_rerank([DummyNode(0.1)], requested=False) is False


def test_should_apply_rerank_false_when_no_nodes():
    mgr = RerankerManager()
    assert mgr.should_apply_rerank([], requested=True) is False


def test_should_apply_rerank_false_when_no_cohere_key(monkeypatch):
    # available depends on get_cohere_key() from rag.config
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: None)

    mgr = reranker_mod.RerankerManager()
    assert mgr.available is False
    assert mgr.should_apply_rerank([DummyNode(0.2), DummyNode(0.1), DummyNode(0.0)]) is False


def test_should_apply_rerank_false_when_too_few_results(monkeypatch):
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "key")
    monkeypatch.setattr(reranker_mod.settings, "rerank_min_results", 3)

    mgr = reranker_mod.RerankerManager()
    assert mgr.should_apply_rerank([DummyNode(0.2), DummyNode(0.1)]) is False


def test_should_apply_rerank_true_when_scores_close(monkeypatch):
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "key")
    monkeypatch.setattr(reranker_mod.settings, "rerank_min_results", 3)
    monkeypatch.setattr(reranker_mod.settings, "rerank_delta_threshold", 0.05)

    mgr = reranker_mod.RerankerManager()

    # delta 0.01 < 0.05 -> rerank
    nodes = [DummyNode(0.50), DummyNode(0.49), DummyNode(0.10)]
    assert mgr.should_apply_rerank(nodes, requested=True) is True


def test_should_apply_rerank_false_when_clear_winner(monkeypatch):
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "key")
    monkeypatch.setattr(reranker_mod.settings, "rerank_min_results", 3)
    monkeypatch.setattr(reranker_mod.settings, "rerank_delta_threshold", 0.05)

    mgr = reranker_mod.RerankerManager()

    # delta 0.2 >= 0.05 -> skip rerank
    nodes = [DummyNode(0.80), DummyNode(0.60), DummyNode(0.10)]
    assert mgr.should_apply_rerank(nodes, requested=True) is False


def test_should_apply_rerank_false_when_missing_scores_but_enough_present(monkeypatch):
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "key")
    monkeypatch.setattr(reranker_mod.settings, "rerank_min_results", 3)
    monkeypatch.setattr(reranker_mod.settings, "rerank_delta_threshold", 0.05)

    mgr = reranker_mod.RerankerManager()

    # Two non-None scores are present; delta >= threshold -> skip rerank
    nodes = [DummyNode(None), DummyNode(0.30), DummyNode(0.10)]
    assert mgr.should_apply_rerank(nodes, requested=True) is False


def test_should_apply_rerank_true_when_only_one_non_none_score(monkeypatch):
    import rag.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_cohere_key", lambda: "key")
    monkeypatch.setattr(reranker_mod.settings, "rerank_min_results", 3)

    mgr = reranker_mod.RerankerManager()

    # Fewer than 2 non-None scores -> rerank
    nodes = [DummyNode(None), DummyNode(0.2), DummyNode(None)]
    assert mgr.should_apply_rerank(nodes, requested=True) is True
