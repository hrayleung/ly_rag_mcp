"""
Storage layer for the RAG system.

Manages ChromaDB connections and LlamaIndex storage.
"""

# Lazy imports to avoid dependency issues at import time
def __getattr__(name):
    if name == "ChromaManager":
        from rag.storage.chroma import ChromaManager
        return ChromaManager
    elif name == "IndexManager":
        from rag.storage.index import IndexManager
        return IndexManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ChromaManager", "IndexManager"]
