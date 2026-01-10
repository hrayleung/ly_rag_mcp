"""Small unit tests for rag.storage.index IndexManager locking and caching."""

from types import SimpleNamespace
import threading

import pytest

import rag.storage.index as index_mod


class DummyChroma:
    def __init__(self):
        self.calls = []

    def get_client(self, project):
        self.calls.append(project)
        return None, SimpleNamespace()

    def reset(self):
        self.calls.append("reset")


class DummyVectorStoreIndex:
    def __init__(self):
        self.storage_context = SimpleNamespace(persist=lambda persist_dir: None)

    def as_retriever(self, similarity_top_k):
        return ("retriever", similarity_top_k)

    def insert_nodes(self, nodes, show_progress=False):
        return len(nodes)


@pytest.fixture(autouse=True)
def patch_llama(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    # Avoid real embedding/model resolution
    monkeypatch.setattr(index_mod, "get_embedding_model", lambda: "default")
    monkeypatch.setattr(index_mod, "LlamaSettings", SimpleNamespace(embed_model=None, text_splitter=None))
    monkeypatch.setattr(index_mod, "VectorStoreIndex", lambda *a, **k: DummyVectorStoreIndex())
    monkeypatch.setattr(index_mod, "load_index_from_storage", lambda sc: DummyVectorStoreIndex())
    monkeypatch.setattr(index_mod, "ChromaVectorStore", lambda chroma_collection: SimpleNamespace())
    monkeypatch.setattr(index_mod, "SentenceSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace())


def test_get_index_caches_and_tracks_stats(monkeypatch, tmp_path):
    chroma = DummyChroma()
    monkeypatch.setattr(index_mod, "get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(index_mod.settings, "storage_path", tmp_path)
    monkeypatch.setattr(index_mod.settings, "default_project", "p")

    mgr = index_mod.IndexManager()
    idx1 = mgr.get_index("p")
    assert mgr.stats.index_loads == 1
    idx2 = mgr.get_index("p")
    assert idx1 is idx2
    assert mgr.stats.index_cache_hits == 1


def test_switch_project_resets(monkeypatch, tmp_path):
    chroma = DummyChroma()
    monkeypatch.setattr(index_mod, "get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(index_mod.settings, "storage_path", tmp_path)
    monkeypatch.setattr(index_mod.settings, "default_project", "p")

    mgr = index_mod.IndexManager()
    mgr.get_index("p")
    mgr.switch_project("q")
    assert chroma.calls[:2] == ["p", "reset"]
    assert mgr.current_project == "q"


def test_get_retriever_uses_top_k(monkeypatch, tmp_path):
    chroma = DummyChroma()
    monkeypatch.setattr(index_mod, "get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(index_mod.settings, "storage_path", tmp_path)
    monkeypatch.setattr(index_mod.settings, "default_project", "p")

    mgr = index_mod.IndexManager()
    retr = mgr.get_retriever(project="p", similarity_top_k=4)
    assert retr == ("retriever", 4)


def test_thread_safety_of_get_index(monkeypatch, tmp_path):
    chroma = DummyChroma()
    monkeypatch.setattr(index_mod, "get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(index_mod.settings, "storage_path", tmp_path)
    monkeypatch.setattr(index_mod.settings, "default_project", "p")

    mgr = index_mod.IndexManager()
    errors = []

    def worker():
        try:
            mgr.get_index("p")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert mgr.stats.index_loads >= 1


def test_get_retriever_log_messages(monkeypatch, tmp_path, caplog):
    """Test that get_retriever logs accurate warnings with original values."""
    import logging

    chroma = DummyChroma()
    monkeypatch.setattr(index_mod, "get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(index_mod.settings, "storage_path", tmp_path)
    monkeypatch.setattr(index_mod.settings, "default_project", "p")
    monkeypatch.setattr(index_mod.settings, "min_top_k", 2)
    monkeypatch.setattr(index_mod.settings, "max_top_k", 20)

    mgr = index_mod.IndexManager()

    # Test log message when top_k too small
    with caplog.at_level(logging.WARNING):
        retr = mgr.get_retriever(project="p", similarity_top_k=1)
        assert retr == ("retriever", 2)  # Clamped to min_top_k

    # Check that warning shows original value
    assert any("top_k=1 too small" in record.message for record in caplog.records)
    caplog.clear()

    # Test log message when top_k too large
    with caplog.at_level(logging.WARNING):
        retr = mgr.get_retriever(project="p", similarity_top_k=50)
        assert retr == ("retriever", 20)  # Clamped to max_top_k

    # Check that warning shows original value
    assert any("top_k=50 too large" in record.message for record in caplog.records)

