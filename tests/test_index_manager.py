"""
Tests for rag.storage.index.IndexManager locking and project switching.
"""

import threading
from unittest.mock import patch, MagicMock

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
