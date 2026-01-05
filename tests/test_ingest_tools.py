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
    """Register tool and inject fakes via class patches (instances now created inside functions)."""
    mcp = DummyMCP()
    register_ingest_tools(mcp)
    tool = mcp.tools["index_documents"]

    # Create default fakes if not provided
    fake_loader = loader if loader else MagicMock()
    if not loader:
        fake_loader.load_directory = MagicMock(return_value=([], {}))
    fake_processor = processor if processor else MagicMock()
    if not processor:
        fake_processor.sanitize_metadata = MagicMock(side_effect=lambda m: m)
        fake_processor.inject_context = MagicMock(side_effect=lambda d: d)
    fake_chunker = chunker if chunker else MagicMock()
    if not chunker:
        fake_chunker.chunk_document = MagicMock(return_value=[])

    # Default manager mocks if not provided
    fake_pm = pm if pm else MagicMock()
    if not isinstance(fake_pm.discover_projects.return_value, list):
        fake_pm.discover_projects.return_value = []
    fake_mm = mm if mm else MagicMock()
    fake_im = im if im else MagicMock()
    fake_im.insert_nodes.return_value = 0

    def wrapped(*args, **kwargs):
        with patch("rag.tools.ingest.DocumentLoader", return_value=fake_loader), \
             patch("rag.tools.ingest.DocumentProcessor", return_value=fake_processor), \
             patch("rag.tools.ingest.DocumentChunker", return_value=fake_chunker), \
             patch("rag.tools.ingest.get_project_manager", return_value=fake_pm), \
             patch("rag.tools.ingest.get_metadata_manager", return_value=fake_mm), \
             patch("rag.tools.ingest.get_index_manager", return_value=fake_im):
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


def test_index_documents_happy_path(tmp_path):
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
    assert resp.get("documents_processed") == 1
    assert resp.get("chunks_created") == 1
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


def test_file_size_checked_early(tmp_path):
    """Bug L7: Verify file size is checked BEFORE reading file contents."""
    from rag.ingestion.loader import DocumentLoader
    from rag.config import settings

    # Create a large file that exceeds the limit
    large_file = tmp_path / "large.txt"
    # Write data larger than max_file_size_bytes
    large_file.write_text("a" * (settings.max_file_size_bytes + 1024))

    loader = DocumentLoader()

    # Validate should reject the file without reading its contents
    is_valid, reason = loader.validate_file(large_file)

    assert is_valid is False
    assert reason == "file_too_large"


def test_file_size_accepts_small_files(tmp_path):
    """Bug L7: Verify files under size limit are accepted."""
    from rag.ingestion.loader import DocumentLoader
    from rag.config import settings

    # Create a small file
    small_file = tmp_path / "small.txt"
    small_file.write_text("hello world")

    loader = DocumentLoader()

    # Validate should accept the file
    is_valid, reason = loader.validate_file(small_file)

    assert is_valid is True
    assert reason == ""


def test_add_text_sanitization():
    """Bug L5: Verify add_text sanitizes input."""
    from rag.tools.ingest import sanitize_text_input

    # Test with leading/trailing whitespace
    text, error = sanitize_text_input("   test content   ")
    assert error is None
    assert text == "test content"

    # Test with only whitespace
    text, error = sanitize_text_input("   \n\t   ")
    assert error is not None
    assert "empty" in error.lower()

    # Test with text too long
    text, error = sanitize_text_input("a" * 100001, max_length=100000)
    assert error is not None
    assert "too long" in error.lower()


def test_sanitize_path_validation():
    """Bug L5: Verify sanitize_path validates paths."""
    from rag.tools.ingest import sanitize_path

    # Test with whitespace
    path, error = sanitize_path("  /tmp/test  ")
    # Should fail because path doesn't exist, not because of whitespace
    assert error is not None
    assert "not found" in error.lower()

    # Test with empty string
    path, error = sanitize_path("")
    assert error is not None
    assert "empty" in error.lower()

    # Test with None
    path, error = sanitize_path(None)
    assert error is not None

