"""
RAG (Retrieval-Augmented Generation) Package

A modular RAG system built with LlamaIndex, ChromaDB, and FastMCP.
"""

__version__ = "2.0.0"

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "settings":
        from rag.config import settings
        return settings
    elif name == "IndexManager":
        from rag.storage.index import IndexManager
        return IndexManager
    elif name == "ProjectManager":
        from rag.project.manager import ProjectManager
        return ProjectManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "settings",
    "IndexManager",
    "ProjectManager",
]
