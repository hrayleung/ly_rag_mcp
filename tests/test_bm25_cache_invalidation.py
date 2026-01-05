"""
Test BM25 cache invalidation with content changes.
"""

import pytest
from types import SimpleNamespace

from rag.retrieval.bm25 import BM25Manager
from llama_index.core.schema import TextNode


def test_compute_content_hash():
    """Test content hash computation."""
    manager = BM25Manager()

    # Create TextNode objects
    node1 = TextNode(
        text="This is test content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    node2 = TextNode(
        text="Another test content",
        id_="node2",
        metadata={"file_path": "/test/file2.py", "mtime_ns": 67890}
    )

    nodes = [node1, node2]

    # Compute hash
    hash1 = manager._compute_content_hash(nodes)

    # Verify hash is non-empty and consistent
    assert hash1
    assert len(hash1) == 32  # MD5 hash length

    # Same nodes should produce same hash
    hash2 = manager._compute_content_hash(nodes)
    assert hash1 == hash2

    # Different content should produce different hash
    node1.text = "Modified content"
    hash3 = manager._compute_content_hash(nodes)
    assert hash1 != hash3


def test_cache_logic_with_content_change():
    """Test cache invalidation logic when content changes."""
    manager = BM25Manager()

    # Create TextNode objects
    node1 = TextNode(
        text="Original content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    nodes = [node1]
    hash1 = manager._compute_content_hash(nodes)

    # Simulate cache state
    manager._retriever = "mock_retriever"
    manager._doc_count_at_build = len(nodes)
    manager._content_hash_at_build = hash1

    # Same content should hit cache
    current_hash = manager._compute_content_hash(nodes)
    cache_valid = (
        manager._retriever is not None
        and manager._doc_count_at_build == len(nodes)
        and manager._content_hash_at_build == current_hash
    )
    assert cache_valid

    # Modify content
    node1.text = "Modified content"
    hash2 = manager._compute_content_hash(nodes)

    # Cache should be invalidated
    cache_valid = (
        manager._retriever is not None
        and manager._doc_count_at_build == len(nodes)
        and manager._content_hash_at_build == hash2
    )
    assert not cache_valid
    assert hash1 != hash2


def test_cache_invalidated_with_mtime_change():
    """Test cache invalidated when mtime changes (file edited)."""
    manager = BM25Manager()

    # Create TextNode with stable content
    node1 = TextNode(
        text="Stable content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    nodes = [node1]
    hash1 = manager._compute_content_hash(nodes)

    # Simulate cache state
    manager._retriever = "mock_retriever"
    manager._doc_count_at_build = len(nodes)
    manager._content_hash_at_build = hash1

    # Update mtime (simulating file edit) but keep content same
    node1.metadata["mtime_ns"] = 99999
    hash2 = manager._compute_content_hash(nodes)

    # Cache should be invalidated (mtime change indicates file was edited)
    cache_valid = (
        manager._retriever is not None
        and manager._doc_count_at_build == len(nodes)
        and manager._content_hash_at_build == hash2
    )
    assert not cache_valid
    assert hash1 != hash2
    # Note: mtime is included in hash to detect file edits


def test_cache_invalidation_logs_on_hash_change(monkeypatch, caplog):
    manager = BM25Manager()

    node = TextNode(
        text="Original",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 1},
    )

    class FakeDocstore:
        def __init__(self, docs):
            self.docs = docs

    class FakeIndex:
        def __init__(self, docs):
            self.docstore = FakeDocstore(docs)

    # Stub BM25 retriever build to avoid heavy dependency
    def fake_from_defaults(nodes, similarity_top_k, stemmer, language):  # noqa: D401
        return "bm25"

    import rag.retrieval.bm25 as bm25_mod
    monkeypatch.setattr(bm25_mod, "BM25Retriever", SimpleNamespace(from_defaults=fake_from_defaults))

    index = FakeIndex({"1": node})
    caplog.set_level("INFO")

    first = manager.get_retriever(index)
    assert first == "bm25"

    # Mutate content but keep count same to force invalidation path
    index.docstore.docs["1"].text = "Changed"
    second = manager.get_retriever(index)

    assert second == "bm25"
    assert manager._stats.bm25_builds == 2
    assert any("cache invalidated" in rec.message for rec in caplog.records)


[["test_cache_reset_clears_hash()"]]


def test_empty_nodes_hash():
    """Test hash computation with empty nodes list."""
    manager = BM25Manager()
    hash_val = manager._compute_content_hash([])
    assert hash_val == ""


def test_nodes_without_metadata():
    """Test hash computation with nodes that have no metadata."""
    manager = BM25Manager()

    # Create node without file_path/mtime
    node = TextNode(text="Content only", id_="node1")
    hash1 = manager._compute_content_hash([node])

    # Should still compute a hash based on text
    assert hash1
    assert len(hash1) == 32

    # Same content should produce same hash
    hash2 = manager._compute_content_hash([node])
    assert hash1 == hash2


def test_hash_includes_file_metadata():
    """Test that hash includes file path and mtime when present."""
    manager = BM25Manager()

    node1 = TextNode(
        text="Same content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    node2 = TextNode(
        text="Same content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 99999}  # Different mtime
    )

    # Different mtime should produce different hash
    hash1 = manager._compute_content_hash([node1])
    hash2 = manager._compute_content_hash([node2])
    assert hash1 != hash2

    # Same file path and mtime should produce same hash
    node3 = TextNode(
        text="Same content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )
    hash3 = manager._compute_content_hash([node3])
    assert hash1 == hash3


def test_bm25_cache_thread_safe():
    """
    Test BM25 cache validation is thread-safe under concurrent access.

    This test verifies Bug H3 fix: nodes access and hash computation
    now happen INSIDE the lock, preventing TOCTOU vulnerability.
    """
    import threading
    import time

    manager = BM25Manager()

    # Create TextNode objects
    node1 = TextNode(
        text="Thread-safe test content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    class FakeDocstore:
        def __init__(self, docs):
            self.docs = docs

    class FakeIndex:
        def __init__(self, docs):
            self.docstore = FakeDocstore(docs)

    # Stub BM25 retriever build to avoid heavy dependency
    def fake_from_defaults(nodes, similarity_top_k, stemmer, language):
        return "bm25"

    import rag.retrieval.bm25 as bm25_mod
    original_retriever = bm25_mod.BM25Retriever
    bm25_mod.BM25Retriever = SimpleNamespace(from_defaults=fake_from_defaults)

    try:
        index = FakeIndex({"1": node1})

        # Track successful retrievals
        results = []
        errors = []

        def concurrent_get():
            try:
                for _ in range(10):
                    retriever = manager.get_retriever(index)
                    results.append(retriever)
                    time.sleep(0.001)  # Small delay to increase race likelihood
            except Exception as e:
                errors.append(e)

        # Launch multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=concurrent_get)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Thread-safe test failed with errors: {errors}"

        # Verify all retrievals succeeded
        assert len(results) == 50  # 5 threads * 10 iterations
        assert all(r == "bm25" for r in results)

        # Verify cache was used (should have built once, then cached)
        # Note: In concurrent scenario, might build multiple times due to races
        # but should be much less than total requests
        assert manager._stats.bm25_builds >= 1
        assert manager._stats.bm25_cache_hits > 0

    finally:
        # Restore original
        bm25_mod.BM25Retriever = original_retriever


def test_cache_reset_clears_hash():
    """Test that reset() clears all cached state including hashes."""
    manager = BM25Manager()

    # Create TextNode objects
    node1 = TextNode(
        text="Test content",
        id_="node1",
        metadata={"file_path": "/test/file1.py", "mtime_ns": 12345}
    )

    class FakeDocstore:
        def __init__(self, docs):
            self.docs = docs

    class FakeIndex:
        def __init__(self, docs):
            self.docstore = FakeDocstore(docs)

    # Stub BM25 retriever build
    def fake_from_defaults(nodes, similarity_top_k, stemmer, language):
        return "bm25"

    import rag.retrieval.bm25 as bm25_mod
    original_retriever = bm25_mod.BM25Retriever
    bm25_mod.BM25Retriever = SimpleNamespace(from_defaults=fake_from_defaults)

    try:
        index = FakeIndex({"1": node1})

        # First call builds and caches
        first = manager.get_retriever(index, project="test_proj")
        assert first == "bm25"
        assert manager._stats.bm25_builds == 1

        # Verify cache state
        assert "test_proj" in manager._retrievers
        assert "test_proj" in manager._doc_count_at_build
        assert "test_proj" in manager._content_hash_at_build

        # Reset cache
        manager.reset()

        # Verify all state cleared
        assert len(manager._retrievers) == 0
        assert len(manager._doc_count_at_build) == 0
        assert len(manager._content_hash_at_build) == 0

        # Next call should rebuild
        second = manager.get_retriever(index, project="test_proj")
        assert second == "bm25"
        assert manager._stats.bm25_builds == 2  # Rebuilt after reset

    finally:
        # Restore original
        bm25_mod.BM25Retriever = original_retriever


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_mtime_ns_fallback():
    """
    Test mtime_ns platform compatibility with fallback (Bug M6).
    
    Verifies that the DocumentLoader properly handles platforms
    or Python versions that don't support st_mtime_ns.
    """
    from pathlib import Path
    from rag.ingestion.loader import DocumentLoader
    from unittest.mock import Mock, patch
    import tempfile

    loader = DocumentLoader()

    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        # Test 1: Normal platform with st_mtime_ns
        mock_stat = Mock()
        mock_stat.st_mtime_ns = 1234567890000000000
        mock_stat.st_mtime = 1234567890.0

        with patch.object(Path, 'stat', return_value=mock_stat):
            from llama_index.core import Document
            documents = loader.load_files([temp_path])
            
            assert len(documents) > 0
            # Should have mtime_ns from st_mtime_ns
            if 'file_path' in documents[0].metadata:
                assert 'mtime_ns' in documents[0].metadata
                assert documents[0].metadata['mtime_ns'] == 1234567890000000000

        # Test 2: Platform without st_mtime_ns (old Python)
        mock_stat_no_ns = Mock()
        mock_stat_no_ns.st_mtime = 1234567890.0
        # Simulate AttributeError when accessing st_mtime_ns
        del mock_stat_no_ns.st_mtime_ns

        with patch.object(Path, 'stat', return_value=mock_stat_no_ns):
            documents = loader.load_files([temp_path])
            
            assert len(documents) > 0
            # Should have mtime_ns from fallback calculation
            if 'file_path' in documents[0].metadata:
                assert 'mtime_ns' in documents[0].metadata
                # Should be calculated from st_mtime * 1e9
                assert documents[0].metadata['mtime_ns'] == int(1234567890.0 * 1e9)

    finally:
        # Cleanup
        temp_path.unlink(missing_ok=True)
