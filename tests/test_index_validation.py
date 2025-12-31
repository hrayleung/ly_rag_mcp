"""
Test index health validation.
"""

import pytest
from rag.storage.index import get_index_manager


def test_validate_index_empty():
    """Test validation of non-existent index."""
    manager = get_index_manager()
    result = manager.validate_index("non_existent_project")

    assert not result["healthy"]
    assert not result["storage_exists"]
    assert not result["index_accessible"]
    assert result["node_count"] == 0
    assert result["chroma_count"] == 0
    assert not result["counts_match"]
    assert any("Storage directory not found" in error for error in result["errors"])


def test_validate_index_current():
    """Test validation of current project index."""
    manager = get_index_manager()

    # Test with default project (might not exist)
    result = manager.validate_index()

    # Should not crash and return required fields
    assert "healthy" in result
    assert "node_count" in result
    assert "chroma_count" in result
    assert "storage_exists" in result
    assert "index_accessible" in result
    assert "counts_match" in result
    assert "errors" in result
    assert isinstance(result["errors"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])