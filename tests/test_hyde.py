import types

import pytest

from rag.retrieval.hyde import generate_hyde_query, should_trigger_hyde


class DummyNode:
    def __init__(self, score):
        self.score = score
        self.node = types.SimpleNamespace(get_content=lambda: "", metadata={}, node_id="n1")


def test_generate_hyde_query_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    question = "What is HyDE?"
    assert generate_hyde_query(question) == question


def test_should_trigger_hyde_no_results():
    assert should_trigger_hyde([]) is True


def test_should_trigger_hyde_none_scores():
    nodes = [DummyNode(None), DummyNode(None)]
    assert should_trigger_hyde(nodes) is True


def test_should_trigger_hyde_low_single_score():
    nodes = [DummyNode(0.05)]
    assert should_trigger_hyde(nodes) is True


def test_should_trigger_hyde_all_low_scores():
    nodes = [DummyNode(0.05), DummyNode(0.15)]
    assert should_trigger_hyde(nodes) is True


def test_should_trigger_hyde_not_trigger_when_high_score():
    nodes = [DummyNode(0.21), DummyNode(0.10)]
    assert should_trigger_hyde(nodes) is False
