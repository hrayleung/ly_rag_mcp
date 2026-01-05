"""
Test ChromaManager cache and state management fixes.

Tests for Bug H13: delete_collection() cache inconsistency
Tests for Bug H14: reset() exception handling
"""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

from rag.storage.chroma import ChromaManager


def test_delete_collection_only_resets_current():
    """
    Test that delete_collection() only resets cache when deleting current project.

    Bug H13 fix: verify that deleting a different project's collection
    does NOT reset the current project's cache.
    """
    manager = ChromaManager()

    # Mock client
    mock_client = Mock()
    mock_collection = Mock()

    # Setup: simulate current project is "project_a"
    manager._client = mock_client
    manager._collection = mock_collection
    manager._current_project = "project_a"

    # Mock client.delete_collection
    mock_client.delete_collection = Mock()

    # Test 1: Delete current project - should reset
    with patch.object(manager, 'get_client', return_value=(mock_client, mock_collection)):
        result = manager.delete_collection("project_a")

    assert result is True
    # State should be cleared after deleting current project
    assert manager._client is None
    assert manager._collection is None
    assert manager._current_project is None

    # Restore state for second test
    mock_client2 = Mock()
    mock_collection2 = Mock()
    manager._client = mock_client2
    manager._collection = mock_collection2
    manager._current_project = "project_a"

    # Test 2: Delete DIFFERENT project - should NOT reset
    mock_client2.delete_collection = Mock()

    with patch.object(manager, 'get_client', return_value=(mock_client2, mock_collection2)):
        result = manager.delete_collection("project_b")

    assert result is True
    # State should be PRESERVED when deleting other project
    assert manager._client is mock_client2
    assert manager._collection is mock_collection2
    assert manager._current_project == "project_a"


def test_delete_collection_handles_exception():
    """Test that delete_collection() handles exceptions gracefully."""
    manager = ChromaManager()

    # Mock client that raises exception
    mock_client = Mock()
    mock_client.delete_collection = Mock(side_effect=Exception("Collection not found"))

    # Setup state
    manager._client = mock_client
    manager._collection = Mock()
    manager._current_project = "test_project"

    # Mock get_client to return our mock
    with patch.object(manager, 'get_client', return_value=(mock_client, Mock())):
        result = manager.delete_collection("test_project")

    # Should return False on error
    assert result is False
    # State should still be set (get_client was called, but deletion failed)


def test_chroma_reset_preserves_on_close_error():
    """
    Test that reset() preserves state if client.close() raises exception.

    Bug H14 fix: verify that if close() fails, state is NOT cleared
    to prevent resource leaks and keep references for retry.
    """
    manager = ChromaManager()

    # Create mock client that fails on close
    mock_client = Mock()
    close_error = RuntimeError("Failed to close client connection")
    mock_client.close = Mock(side_effect=close_error)

    mock_collection = Mock()

    # Setup state
    manager._client = mock_client
    manager._collection = mock_collection
    manager._current_project = "test_project"

    # Attempt reset - should raise exception
    with pytest.raises(RuntimeError, match="Failed to close"):
        manager.reset()

    # State should be PRESERVED after failed close
    assert manager._client is mock_client, "Client should be preserved on close failure"
    assert manager._collection is mock_collection, "Collection should be preserved on close failure"
    assert manager._current_project == "test_project", "Project should be preserved on close failure"

    # Verify close was attempted
    mock_client.close.assert_called_once()


def test_chroma_reset_clears_state_on_success():
    """Test that reset() properly clears state when close() succeeds."""
    manager = ChromaManager()

    # Create mock client that succeeds on close
    mock_client = Mock()
    mock_client.close = Mock()  # No exception

    mock_collection = Mock()

    # Setup state
    manager._client = mock_client
    manager._collection = mock_collection
    manager._current_project = "test_project"

    # Reset should succeed
    manager.reset()

    # State should be cleared
    assert manager._client is None
    assert manager._collection is None
    assert manager._current_project is None

    # Verify close was called
    mock_client.close.assert_called_once()


def test_chroma_reset_with_no_client():
    """Test that reset() handles case when no client is set."""
    manager = ChromaManager()

    # No client set
    assert manager._client is None

    # Should not raise exception
    manager.reset()

    # State should remain clear
    assert manager._client is None
    assert manager._collection is None
    assert manager._current_project is None


def test_chroma_reset_with_client_without_close_method():
    """Test that reset() handles client without close() method."""
    manager = ChromaManager()

    # Create mock client without close method
    mock_client = SimpleNamespace()  # No close attribute
    mock_collection = Mock()

    # Setup state
    manager._client = mock_client
    manager._collection = mock_collection
    manager._current_project = "test_project"

    # Should succeed without calling close
    manager.reset()

    # State should be cleared
    assert manager._client is None
    assert manager._collection is None
    assert manager._current_project is None


def test_delete_collection_when_cache_empty():
    """Test delete_collection() when cache is empty (no current project)."""
    manager = ChromaManager()

    # No current project
    assert manager._current_project is None

    # Mock client
    mock_client = Mock()
    mock_client.delete_collection = Mock()

    with patch.object(manager, 'get_client', return_value=(mock_client, Mock())):
        result = manager.delete_collection("some_project")

    # Should succeed
    assert result is True
    # Cache should remain empty (wasn't current project)
    assert manager._current_project is None
    assert manager._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
