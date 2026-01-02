
import pytest
from unittest.mock import MagicMock, patch
from rag.tools.ingest import register_ingest_tools, DocumentChunker
from llama_index.core import Document

class DummyMCP:
    def __init__(self):
        self.tools = {}
    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

@pytest.fixture
def mcp_setup():
    mcp = DummyMCP()
    register_ingest_tools(mcp)
    return mcp

def test_index_documents_partial_failure_crash(tmp_path, mcp_setup):
    """
    Test that index_documents crashes the whole process if one chunk operation fails,
    instead of partially succeeding.
    """
    tool = mcp_setup.tools["index_documents"]

    # Setup mocks
    mock_pm = MagicMock()
    mock_pm.discover_projects.return_value = ["test_project"]
    mock_pm.create_project.return_value = {"success": True}

    mock_im = MagicMock()

    # Create 3 documents
    docs = [
        Document(text="doc1", metadata={"file_path": str(tmp_path / "1.txt")}),
        Document(text="doc2", metadata={"file_path": str(tmp_path / "2.txt")}),
        Document(text="doc3", metadata={"file_path": str(tmp_path / "3.txt")})
    ]

    mock_loader = MagicMock()
    mock_loader.load_directory.return_value = (docs, {})

    # Mock chunker to fail on the second document
    mock_chunker = MagicMock()
    def side_effect(doc):
        if "2.txt" in doc.metadata.get("file_path", ""):
            raise ValueError("Chunking failed for bad document")
        return [MagicMock(text=doc.text)]

    mock_chunker.chunk_document.side_effect = side_effect

    with patch("rag.tools.ingest.get_project_manager", return_value=mock_pm), \
         patch("rag.tools.ingest.get_index_manager", return_value=mock_im), \
         patch("rag.tools.ingest.DocumentLoader", return_value=mock_loader), \
         patch("rag.tools.ingest.DocumentProcessor", return_value=MagicMock(sanitize_metadata=lambda x: x, inject_context=lambda x: x)), \
         patch("rag.tools.ingest.DocumentChunker", return_value=mock_chunker):

        # Test runs here
        dir_path = tmp_path / "test_docs"
        dir_path.mkdir()

        result = tool(path=str(dir_path), project="test_project")

        # We expect the whole operation to return an error because the exception isn't caught inside the loop
        assert "error" in result
        assert "Chunking failed for bad document" in result["error"]

        # Verify that insert_nodes was NOT called (meaning nothing was indexed)
        # OR verify that it wasn't called for the valid documents (total failure)
        mock_im.insert_nodes.assert_not_called()

def test_add_text_sanitization_missing(mcp_setup):
    """
    Test that add_text does not sanitize input text.
    """
    tool = mcp_setup.tools["add_text"]

    mock_im = MagicMock()
    mock_index = MagicMock()
    mock_im.get_index.return_value = mock_index

    dirty_text = "Hello\x00World" # Null byte

    with patch("rag.tools.ingest.get_index_manager", return_value=mock_im):
        tool(text=dirty_text, project="test")

        # Get the nodes passed to insert_nodes
        args, _ = mock_index.insert_nodes.call_args
        nodes = args[0]
        text_processed = nodes[0].text

        # If sanitization was skipped, the null byte will still be there
        # If it was sanitized, it would handle it (DocumentProcessor.clean_text handles it?)
        # Actually DocumentProcessor.clean_text removes non-printable. \x00 is non-printable.
        # So if it fails validation, text_processed == dirty_text.

        # Note: clean_text in processor.py:
        # cleaned = ''.join(c for c in cleaned if c.isprintable() or c in '\n\t ')

        assert "\x00" in text_processed
