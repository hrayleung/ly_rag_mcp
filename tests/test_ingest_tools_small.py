"""Fast coverage for rag.tools.ingest index_documents/add_text/inspect_directory."""

from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

import rag.tools.ingest as ingest_mod


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


class DummyMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _register(monkeypatch, pm=None, im=None, mm=None, loader=None, processor=None, chunker=None, settings=None):
    if pm is None:
        pm = SimpleNamespace(
            discover_projects=lambda: [],
            create_project=lambda project: {"ok": True},
            set_project_metadata=lambda project, keywords=None: None,
            update_project_paths=lambda project, paths: None,
        )
    # normalize create_project to return dict without probing side effects
    if pm:
        original_pm = pm
        def _wrap_create(name):
            result = original_pm.create_project(name)
            if isinstance(result, dict):
                return result
            return {"ok": True}
        pm = SimpleNamespace(
            discover_projects=original_pm.discover_projects,
            create_project=_wrap_create,
            set_project_metadata=original_pm.set_project_metadata,
            update_project_paths=original_pm.update_project_paths,
        )
    if im is None:
        im = SimpleNamespace(
            switch_project=lambda project: None,
            insert_nodes=lambda nodes, project, show_progress=False: len(nodes),
            persist=lambda project: None,
            get_index=lambda project=None: SimpleNamespace(insert_nodes=lambda n, show_progress=False: len(n)),
            current_project="p",
        )
    # ensure get_index returns object with insert_nodes
    if not hasattr(im.get_index(), "insert_nodes"):
        def _idx(project=None):
            return SimpleNamespace(insert_nodes=lambda n, show_progress=False: len(n))
        im = SimpleNamespace(
            switch_project=im.switch_project,
            insert_nodes=im.insert_nodes,
            persist=im.persist,
            get_index=_idx,
            current_project=getattr(im, "current_project", "p"),
        )
    if mm is None:
        mm = SimpleNamespace(
            extract_keywords_from_directory=lambda dir_path, max_keywords=8: ["k1"],
            record_index_activity=lambda project, source: None,
        )
    if loader is None:
        loader = SimpleNamespace(load_directory=lambda directory: ([SimpleNamespace(text="t", metadata={})], {}))
    if processor is None:
        processor = SimpleNamespace(
            sanitize_metadata=lambda md: md,
            inject_context=lambda doc: doc,
        )
    if chunker is None:
        chunker = SimpleNamespace(chunk_document=lambda doc: ["n1", "n2"])
    if settings is None:
        settings = ingest_mod.settings

    monkeypatch.setattr(ingest_mod, "DocumentLoader", lambda: loader)
    monkeypatch.setattr(ingest_mod, "DocumentProcessor", lambda: processor)
    monkeypatch.setattr(ingest_mod, "DocumentChunker", lambda: chunker)
    monkeypatch.setattr(ingest_mod, "get_project_manager", lambda: pm)
    monkeypatch.setattr(ingest_mod, "get_index_manager", lambda: im)
    monkeypatch.setattr(ingest_mod, "get_metadata_manager", lambda: mm)
    monkeypatch.setattr(ingest_mod, "settings", settings)

    mcp = DummyMCP()
    ingest_mod.register_ingest_tools(mcp)
    return mcp


def test_index_documents_requires_dir(monkeypatch, tmp_path):
    mcp = _register(monkeypatch)
    resp = mcp.tools["index_documents"](path=str(tmp_path / "missing"))
    assert "Path not found" in resp["error"]


def test_index_documents_creates_project_and_indexes(monkeypatch, tmp_path):
    created = {}
    pm = SimpleNamespace(
        discover_projects=lambda: [],
        create_project=lambda project: created.setdefault("name", project),
        set_project_metadata=lambda project, keywords=None: None,
        update_project_paths=lambda project, paths: created.setdefault("paths", paths),
    )
    # create a real dir with a file so loader receives existing path
    dir_path = tmp_path / "d"
    dir_path.mkdir()
    (dir_path / "f.txt").write_text("hello")

    mcp = _register(monkeypatch, pm=pm)
    resp = mcp.tools["index_documents"](path=str(dir_path), project="p1")
    assert resp.get("success") is True
    assert created["name"] == "p1"
    assert "error" not in resp
    assert resp.get("mode") in ("hybrid", "docs")
    assert resp.get("documents_processed") == 1
    assert resp.get("chunks_created") == 2
    assert resp.get("project") == "p1"
    assert created["paths"] == [str(dir_path)]
    assert resp["chunks_created"] >= resp["documents_processed"]
    assert created["paths"] == [str(dir_path)]
    assert resp["documents_processed"] == 1
    assert resp["chunks_created"] == 2
    assert resp["mode"] in ("hybrid", "docs")
    assert resp["project"] == "p1"
    assert "detected_keywords" not in resp
    assert resp["chunks_created"] >= resp["documents_processed"]
    assert "error" not in resp


def test_index_documents_returns_action_when_no_project(monkeypatch, tmp_path):
    dir_path = tmp_path / "d2"
    dir_path.mkdir()
    (dir_path / "f.txt").write_text("hello")
    mcp = _register(monkeypatch)

    resp = mcp.tools["index_documents"](path=str(dir_path), project=None)
    assert resp["action_required"] == "select_project"
    assert "detected_keywords" in resp


def test_add_text_validates(monkeypatch):
    mcp = _register(monkeypatch)
    resp = mcp.tools["add_text"](text=" ")
    assert "error" in resp

    resp2 = mcp.tools["add_text"](text="hello", project="p")
    assert resp2.get("success") is True
    assert resp2["chunks_created"] >= 1
    assert resp2["text_length"] == len("hello")
    assert "error" not in resp2
    assert isinstance(resp2["chunks_created"], int)
    assert isinstance(resp2["text_length"], int)
    assert resp2["text_length"] > 0
    assert resp2["chunks_created"] > 0
    assert resp2.get("project") in (None, "p")


def test_inspect_directory_counts(monkeypatch, tmp_path):
    dir_path = tmp_path / "d3"
    dir_path.mkdir()
    (dir_path / "a.py").write_text("print(1)")
    (dir_path / "b.md").write_text("doc")

    mcp = _register(monkeypatch)
    stats = mcp.tools["inspect_directory"](path=str(dir_path))
    assert stats["code_files"] == 1
    assert stats["doc_files"] == 1
    assert stats["suggested_name"] == "d3"
