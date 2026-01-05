"""Coverage for rag.tools.query routing and formatting edge cases."""

from types import SimpleNamespace
import pytest

import rag.tools.query as query_mod

class DummyMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            # register and also expose as attribute for direct calls
            self.tools[fn.__name__] = fn
            setattr(self, fn.__name__, fn)
            return fn
        return decorator


class FakeResult:
    def __init__(self, results=None, total=0, search_mode="semantic", reranked=False, used_hyde=False):
        self.results = results or []
        self.total = total
        self.search_mode = SimpleNamespace(value=search_mode)
        self.reranked = reranked
        self.used_hyde = used_hyde

class FakeDoc:
    def __init__(self, text, score=None, metadata=None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


def _register(monkeypatch, pm=None, mm=None, engine=None):
    if pm is None:
        pm = SimpleNamespace(
            discover_projects=lambda: ["p1", "p2"],
            choose_project=lambda q, max_candidates=2: {"candidates": [{"project": "p1"}, {"project": "p2"}]},
        )
    if mm is None:
        mm = SimpleNamespace(
            learn_from_query=lambda project, question, success=True: None,
            track_chat_turn=lambda project: None
        )
    if engine is None:
        engine = SimpleNamespace(
            search=lambda question, project, **kwargs: FakeResult(
                results=[FakeDoc("t1", score=None), FakeDoc("t2", score=0.6)],
                total=2,
                search_mode="hybrid",
                reranked=True,
            )
        )

    monkeypatch.setattr(query_mod, "get_project_manager", lambda: pm)
    monkeypatch.setattr(query_mod, "get_metadata_manager", lambda: mm)
    monkeypatch.setattr(query_mod, "get_search_engine", lambda: engine)

    mcp = DummyMCP()
    query_mod.register_query_tools(mcp)
    return mcp


def test_format_result_metadata_and_text(monkeypatch):
    mcp = _register(monkeypatch)
    res = FakeResult(results=[FakeDoc("t", score=None, metadata={"file": "f"})], total=1)
    md = query_mod._format_result(res, "p1", False, True, tried_projects=["p1"], note="n")
    assert md["sources"][0]["score"] is None
    assert md["auto_routed"] is False
    assert md["note"] == "n"

    txt = query_mod._format_result(res, "p1", True, False, tried_projects=["p1", "p2"], note="n")
    assert "Found 1 documents" in txt
    assert "HyDE" not in txt
    assert "tried: p1 → p2" in txt


def test_query_rag_single_project(monkeypatch):
    pm = SimpleNamespace(discover_projects=lambda: ["only"], choose_project=lambda q, max_candidates=1: {})
    engine = SimpleNamespace(search=lambda question, project, **kwargs: FakeResult(results=[FakeDoc("t", score=0.9)], total=1))
    mm = SimpleNamespace(
        learn_from_query=lambda project, question, success=True: None,
        track_chat_turn=lambda project: None
    )
    mcp = _register(monkeypatch, pm=pm, mm=mm, engine=engine)

    out = mcp.query_rag("q")
    assert "Found 1 documents" in out


def test_query_rag_auto_routes_best_effort(monkeypatch):
    calls = []
    def search_fn(question, project, **kwargs):
        calls.append(project)
        if project == "p1":
            return FakeResult(results=[FakeDoc("t", score=0.1)], total=1)
        return FakeResult(results=[FakeDoc("t", score=0.6)], total=1)

    engine = SimpleNamespace(search=search_fn)
    pm = SimpleNamespace(
        discover_projects=lambda: ["p1", "p2"],
        choose_project=lambda q, max_candidates=2: {"candidates": [{"project": "p1"}, {"project": "p2"}]},
    )
    mm = SimpleNamespace(
        learn_from_query=lambda project, question, success=True: None,
        track_chat_turn=lambda project: None
    )
    mcp = _register(monkeypatch, pm=pm, mm=mm, engine=engine)

    out = mcp.query_rag("q")
    assert "Found" in out
    assert calls == ["p1", "p2"]


def test_query_rag_returns_error_no_projects(monkeypatch):
    pm = SimpleNamespace(discover_projects=lambda: [], choose_project=lambda q, max_candidates=0: {})
    mcp = _register(monkeypatch, pm=pm)
    out = mcp.query_rag("q")
    assert "No projects" in out
