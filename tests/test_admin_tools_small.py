"""Lightweight coverage for rag.tools.admin manage_project and get_stats."""

from types import SimpleNamespace
import pytest

import rag.tools.admin as admin_mod


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _register(monkeypatch, pm=None, im=None, cm=None, settings=None):
    if pm is None:
        pm = SimpleNamespace(
            list_projects=lambda: {"details": [{"name": "p"}], "count": 1},
            create_project=lambda project: None,
            set_project_metadata=lambda project, keywords, description: None,
            project_exists=lambda name: name == "p",
            validate_name=lambda name: (True, ""),
            choose_project=lambda desc, max_candidates=3: {
                "recommendation": "p",
                "candidates": [{"project": "p"}],
            },
        )
    if im is None:
        im = SimpleNamespace(current_project="p", stats=SimpleNamespace(index_loads=0, index_cache_hits=0, cache_hit_rate=0.5), switch_project=lambda project: None)
    if cm is None:
        cm = SimpleNamespace(get_collection_count=lambda project=None: 3)
    if settings is None:
        settings = SimpleNamespace(storage_path=SimpleNamespace(exists=lambda: True), embedding_provider="openai", embedding_model="text-embedding-3-large")

    monkeypatch.setattr(admin_mod, "get_project_manager", lambda: pm)
    monkeypatch.setattr(admin_mod, "get_index_manager", lambda: im)
    monkeypatch.setattr(admin_mod, "get_chroma_manager", lambda: cm)
    monkeypatch.setattr(admin_mod, "settings", settings)

    mcp = FakeMCP()
    admin_mod.register_admin_tools(mcp)
    return mcp


def test_manage_project_list(monkeypatch):
    mcp = _register(monkeypatch)
    resp = mcp.tools["manage_project"]("list")
    assert resp["current"] == "p"
    assert resp["total"] == 1


def test_manage_project_create_update_switch(monkeypatch):
    created = {}
    switched = {}

    def create_project(name):
        created["name"] = name

    def switch_project(name):
        switched["name"] = name

    pm = SimpleNamespace(
        list_projects=lambda: {"details": [], "count": 0},
        create_project=create_project,
        set_project_metadata=lambda project, keywords, description: (project, keywords, description),
        project_exists=lambda name: True, # Assume exists for test
        validate_name=lambda name: (True, ""),
        choose_project=lambda desc, max_candidates=3: {"error": "no_projects"},
    )
    im = SimpleNamespace(current_project=None, stats=SimpleNamespace(index_loads=0, index_cache_hits=0, cache_hit_rate=0.5), switch_project=switch_project)
    mcp = _register(monkeypatch, pm=pm, im=im)

    assert mcp.tools["manage_project"]("create", project="x") == {"success": "Created project: x", "next_step": "Add keywords: manage_project(action='update', project='x', keywords=['term1', 'term2'])"}
    assert created["name"] == "x"

    resp_update = mcp.tools["manage_project"]("update", project="x", keywords=["k"], description="d")
    assert resp_update["keywords"] == ["k"]

    resp_switch = mcp.tools["manage_project"]("switch", project="x")
    assert resp_switch["success"] == "Switched to: x"
    assert switched["name"] == "x"


def test_manage_project_analyze_and_unknown(monkeypatch):
    pm = SimpleNamespace(
        list_projects=lambda: {},
        create_project=lambda project: None,
        set_project_metadata=lambda project, keywords, description: None,
        choose_project=lambda desc, max_candidates=3: {"recommendation": "p", "candidates": [{"project": "p"}]},
    )
    im = SimpleNamespace(current_project="p", stats=SimpleNamespace(index_loads=0, index_cache_hits=0, cache_hit_rate=0.5), switch_project=lambda project: None)
    mcp = _register(monkeypatch, pm=pm, im=im)

    analyze = mcp.tools["manage_project"]("analyze", description="how to api")
    assert analyze["recommendation"] == "p"

    unknown = mcp.tools["manage_project"]("noop")
    assert unknown["error"].startswith("Unknown action")


def test_get_stats_cache_and_index(monkeypatch):
    settings = SimpleNamespace(storage_path=SimpleNamespace(exists=lambda: True), embedding_provider="openai", embedding_model="text-embedding-3-large")
    im = SimpleNamespace(
        current_project="p",
        stats=SimpleNamespace(index_loads=1, index_cache_hits=2, cache_hit_rate=0.4),
    )
    cm = SimpleNamespace(get_collection_count=lambda project=None: 7)
    pm = SimpleNamespace()
    mcp = _register(monkeypatch, pm=pm, im=im, cm=cm, settings=settings)

    cache_resp = mcp.tools["get_stats"]("cache")
    assert cache_resp["index_loads"] == 1

    index_resp = mcp.tools["get_stats"]("index")
    assert index_resp["documents"] == 7
    assert index_resp["current_project"] == "p"


def test_list_documents_handles_missing_docstore(monkeypatch):
    settings = SimpleNamespace(storage_path=SimpleNamespace(exists=lambda: True))
    class Index:
        docstore = None
    im = SimpleNamespace(get_index=lambda project=None: Index())
    mcp = _register(monkeypatch, im=im, settings=settings)

    resp = mcp.tools["list_documents"]()
    assert resp["documents"] == []


def test_clear_index_requires_confirm(monkeypatch):
    mcp = _register(monkeypatch)
    resp = mcp.tools["clear_index"](project="p", confirm=False)
    assert "error" in resp
