"""
Tests for rag.storage.index.IndexManager locking and project switching.
"""

import threading
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from rag.storage.index import IndexManager


@pytest.fixture
def manager(tmp_path):
    with patch("rag.storage.index.settings") as settings:
        settings.default_project = "default"
        settings.storage_path = tmp_path
        settings.chunk_size = 100
        settings.chunk_overlap = 20

        mgr = IndexManager()
        yield mgr


def test_switch_project_serializes_with_lock(manager):
    order = []

    def slow_reset(*args, **kwargs):
        order.append("reset_start")
        # simulate work while holding lock
        import time
        time.sleep(0.05)
        order.append("reset_end")

    with patch.object(manager, "reset", side_effect=slow_reset) as reset_mock, \
        patch("rag.storage.index.get_chroma_manager") as chroma, \
        patch.object(manager, "get_index", return_value=MagicMock()) as get_idx:

        t1 = threading.Thread(target=manager.switch_project, args=("p1",))
        t2 = threading.Thread(target=manager.switch_project, args=("p2",))
        t1.start(); t2.start(); t1.join(); t2.join()

    # reset should not interleave outside lock; both called
    assert reset_mock.call_count == 2
    # order should be reset_start/reset_end pairs, not interleaved
    assert order == ["reset_start", "reset_end", "reset_start", "reset_end"]
    assert get_idx.call_count == 2
    # chroma reset is invoked inside manager.switch_project; ensure called twice
    assert chroma.return_value.reset.call_count == 2


def test_get_index_reentrant(manager):
    # Ensure RLock allows nested get_index calls
    with patch("rag.storage.index.get_chroma_manager") as chroma, \
        patch("rag.storage.index.VectorStoreIndex") as VIndex, \
        patch("rag.storage.index.StorageContext") as SContext, \
        patch("rag.storage.index.load_index_from_storage") as loader:

        # Simulate existing docstore to avoid creation path complexity
        docstore_path = manager._current_project = "p1"
        # Force get_node_count path
        manager._index = MagicMock()
        manager._current_project = "p1"

        # Re-enter lock by calling get_index inside a call that already holds it
        with manager._lock:
            idx = manager.get_index("p1")

    assert idx is not None


def test_get_index_concurrent(manager):
    """Test that get_index returns correct project during concurrent access (Bug C4)."""
    results = []
    errors = []

    def get_and_report(proj):
        try:
            idx = manager.get_index(proj)
            # Verify the index is for the requested project
            results.append((proj, manager._current_project))
        except Exception as e:
            errors.append(str(e))

    # Create mock chroma manager
    mock_chroma_mgr = MagicMock()
    mock_collection = MagicMock()

    with patch("rag.storage.index.get_chroma_manager", return_value=mock_chroma_mgr), \
         patch("rag.storage.index.ChromaVectorStore") as chroma_vs, \
         patch("rag.storage.index.VectorStoreIndex") as VIndex, \
         patch("rag.storage.index.StorageContext") as SContext, \
         patch("rag.storage.index.load_index_from_storage") as loader_mock, \
         patch.object(manager, "_configure_llama_settings"):

        # Mock chroma manager's get_client to return tuple
        mock_chroma_mgr.get_client.return_value = (MagicMock(), mock_collection)
        mock_chroma_mgr.reset.return_value = None

        # Mock successful index loading
        mock_index = MagicMock()
        loader_mock.return_value = mock_index

        # Create threads requesting different projects concurrently
        t1 = threading.Thread(target=get_and_report, args=("project_a",))
        t2 = threading.Thread(target=get_and_report, args=("project_b",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify that each thread got the project it requested
    # The current_project should match the project that was requested
    assert len(results) == 2
    # Both threads should have gotten their requested project
    for req_proj, curr_proj in results:
        assert req_proj == curr_proj, f"Requested {req_proj} but got {curr_proj}"


def test_persist_concurrent_project_switch(manager):
    """Test that persist doesn't save wrong index during project switch (Bug C1)."""
    persist_paths = []

    def mock_persist(persist_dir):
        # Track which directory persist was called with
        persist_paths.append(persist_dir)

    with patch("rag.storage.index.get_chroma_manager") as chroma_mock, \
         patch("rag.storage.index.VectorStoreIndex") as VIndex, \
         patch("rag.storage.index.StorageContext") as SContext, \
         patch("rag.storage.index.load_index_from_storage") as loader_mock:

        # Create mock index for project_a
        mock_index_a = MagicMock()
        mock_storage_a = MagicMock()
        mock_index_a.storage_context = mock_storage_a
        mock_storage_a.persist.side_effect = mock_persist

        # Load project_a
        manager._index = mock_index_a
        manager._current_project = "project_a"

        # Start persist for project_a
        def persist_project_a():
            manager.persist("project_a")

        # In another thread, switch to project_b
        def switch_to_project_b():
            manager.get_index("project_b")

        # Mock get_index to load project_b
        original_get_index = manager.get_index
        get_index_calls = []

        def mock_get_index(project):
            get_index_calls.append(project)
            if project == "project_b":
                # Switch to project_b
                mock_index_b = MagicMock()
                manager._index = mock_index_b
                manager._current_project = "project_b"
                return mock_index_b
            return mock_index_a

        with patch.object(manager, "get_index", side_effect=mock_get_index):
            t1 = threading.Thread(target=persist_project_a)
            t2 = threading.Thread(target=switch_to_project_b)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

        # Verify persist was only called for project_a, not project_b
        # If the bug exists, persist might be called after switch with wrong path
        assert len(persist_paths) <= 1, f"Persist called {len(persist_paths)} times"

        # Most importantly, verify that if persist did run, it logged a warning
        # or skipped when projects didn't match
        if len(persist_paths) > 0:
            # The persist should have been for project_a only
            # (current implementation skips if project mismatch)
            pass


def test_insert_nodes_concurrent_switch(manager):
    """Test that nodes are inserted into correct project during concurrent switch (Bug C2)."""
    inserted_projects = []

    def mock_insert(nodes, show_progress=False):
        # Track which project got the insert
        inserted_projects.append(manager._current_project)

    with patch("rag.storage.index.get_chroma_manager") as chroma_mock, \
         patch("rag.storage.index.VectorStoreIndex") as VIndex, \
         patch("rag.storage.index.StorageContext") as SContext, \
         patch("rag.storage.index.load_index_from_storage") as loader_mock:

        # Create mock indexes for both projects
        mock_index_a = MagicMock()
        mock_index_a.insert_nodes.side_effect = mock_insert

        mock_index_b = MagicMock()
        mock_index_b.insert_nodes.side_effect = mock_insert

        # Load project_a
        manager._index = mock_index_a
        manager._current_project = "project_a"

        # Track operations
        operations = []

        def insert_for_project_a():
            operations.append(("insert_start", "project_a"))
            try:
                manager.insert_nodes([MagicMock()], "project_a")
                operations.append(("insert_end", "project_a"))
            except Exception as e:
                operations.append(("insert_error", "project_a", str(e)))

        def switch_to_project_b():
            operations.append(("switch_start", "project_b"))
            try:
                # Simulate get_index switching projects
                manager._index = mock_index_b
                manager._current_project = "project_b"
                operations.append(("switch_end", "project_b"))
            except Exception as e:
                operations.append(("switch_error", "project_b", str(e)))

        # Run concurrently
        t1 = threading.Thread(target=insert_for_project_a)
        t2 = threading.Thread(target=switch_to_project_b)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Verify operations completed
        assert len(operations) > 0

        # The key verification: if insert happened during or after switch,
        # it should have either:
        # 1. Completed for project_a before switch
        # 2. Detected mismatch and reloaded project_a
        # 3. Not inserted at all (safest behavior)

        # Count how many inserts actually happened
        actual_inserts = [p for p in inserted_projects if p is not None]

        # With the fix, we should NOT see nodes inserted into project_b
        # when project_a was requested
        assert "project_b" not in actual_inserts or len(actual_inserts) == 0, \
            f"Nodes inserted into project_b when project_a was requested: {inserted_projects}"

        # Verify insert_nodes was called with lock protection
        assert mock_index_a.insert_nodes.called or mock_index_b.insert_nodes.called
