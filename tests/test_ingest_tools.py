"""
Tests for rag.tools.ingest.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.tools.ingest import register_ingest_tools


class DummyMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


def _setup_index_docs_tool(tmp_path, pm=None, mm=None, im=None, loader=None, processor=None, chunker=None):
    """Register tool and inject fakes via closures and patches."""
    mcp = DummyMCP()
    register_ingest_tools(mcp)
    tool = mcp.tools["index_documents"]

    # Grab closure instances for loader/processor/chunker
    closure_cells = {id(c.cell_contents): c.cell_contents for c in tool.__closure__ or []}
    closure_objs = list(closure_cells.values())

    def pick(attr):
        for obj in closure_objs:
            if hasattr(obj, attr):
                return obj
        return None

    real_loader = pick("load_directory")
    real_processor = pick("sanitize_metadata")
    real_chunker = pick("chunk_document")

    # Override behaviors on captured instances if provided
    if loader and real_loader:
        real_loader.load_directory = loader.load_directory
    if processor and real_processor:
        real_processor.sanitize_metadata = processor.sanitize_metadata
        real_processor.inject_context = processor.inject_context
    if chunker and real_chunker:
        real_chunker.chunk_document = chunker.chunk_document
    # If overrides missing, ensure defaults are harmless no-ops for tests
    if real_loader and not loader:
        real_loader.load_directory = MagicMock(return_value=([], {}))
    if real_processor and not processor:
        real_processor.sanitize_metadata = MagicMock(side_effect=lambda m: m)
        real_processor.inject_context = MagicMock(side_effect=lambda d: d)
    if real_chunker and not chunker:
        real_chunker.chunk_document = MagicMock(return_value=[])

    def wrapped(*args, **kwargs):
        # Default mocks if not provided
        pm_local = pm or MagicMock()
        existing = pm_local.discover_projects.return_value
        if not isinstance(existing, list):
            pm_local.discover_projects.return_value = []
        mm_local = mm or MagicMock()
        im_local = im or MagicMock()
        im_local.insert_nodes.return_value = 0

        with patch("rag.tools.ingest.get_project_manager", return_value=pm_local), \
            patch("rag.tools.ingest.get_metadata_manager", return_value=mm_local), \
            patch("rag.tools.ingest.get_index_manager", return_value=im_local):
            return tool(*args, **kwargs)

    return wrapped



def test_index_documents_returns_error_when_create_project_fails(tmp_path):
    fake_pm = MagicMock()
    fake_pm.discover_projects.return_value = []
    fake_pm.create_project.return_value = {"error": "boom"}

    fake_mm = MagicMock()
    fake_im = MagicMock()
    fake_loader = MagicMock()
    fake_loader.load_directory.side_effect = AssertionError("should not load")
    fake_proc = MagicMock()
    fake_chunker = MagicMock()

    tool = _setup_index_docs_tool(
        tmp_path,
        pm=fake_pm,
        mm=fake_mm,
        im=fake_im,
        loader=fake_loader,
        processor=fake_proc,
        chunker=fake_chunker,
    )

    resp = tool(path=str(tmp_path), project="newproj")

    assert resp == {"error": "boom"}

    # ensure we did not attempt to load documents
    fake_loader.load_directory.assert_not_called()


def test_index_documents_happy_path_without_mode_param(tmp_path):
    fake_pm = MagicMock()
    fake_pm.discover_projects.return_value = []
    fake_pm.create_project.return_value = {"success": True}

    fake_mm = MagicMock()
    fake_mm.extract_keywords_from_directory.return_value = ["k"]

    fake_im = MagicMock()

    doc = MagicMock()
    fake_loader = MagicMock()
    fake_loader.load_directory.return_value = ([doc], {})

    fake_proc = MagicMock()
    fake_proc.sanitize_metadata.side_effect = lambda m: m
    fake_proc.inject_context.side_effect = lambda d: d

    fake_chunker = MagicMock()
    fake_chunker.chunk_document.return_value = [MagicMock()]

    tool = _setup_index_docs_tool(
        tmp_path,
        loader=fake_loader,
        processor=fake_proc,
        chunker=fake_chunker,
        pm=fake_pm,
        mm=fake_mm,
        im=fake_im,
    )

    resp = tool(path=str(tmp_path), project="proj123")

    assert resp.get("success") is True
    assert resp.get("project") == "proj123"
    assert "mode" in resp and resp["mode"] in {"docs", "hybrid"}
    fake_loader.load_directory.assert_called_once()
    fake_chunker.chunk_document.assert_called()
    fake_im.insert_nodes.assert_called()
    fake_im.persist.assert_called()
    fake_mm.record_index_activity.assert_called()
    fake_pm.create_project.assert_called_once_with("proj123")
    fake_im.switch_project.assert_called_once_with("proj123")


def test_index_documents_requires_project_when_missing(tmp_path):
    # no fakes needed beyond pm/mm for prompt path
    # ensure helper works when only pm/mm patching is needed
    fake_pm = MagicMock()
    fake_pm.discover_projects.return_value = ["a", "b"]

    fake_mm = MagicMock()
    fake_mm.extract_keywords_from_directory.return_value = ["kw"]

    tool = _setup_index_docs_tool(tmp_path, pm=fake_pm, mm=fake_mm)

    resp = tool(path=str(tmp_path), project=None)

    assert resp.get("action_required") == "select_project"
    assert "options" in resp
