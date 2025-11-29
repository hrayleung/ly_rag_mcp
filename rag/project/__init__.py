"""
Project management for the RAG system.

Handles multi-project isolation and metadata.
"""

# Lazy imports
def __getattr__(name):
    if name == "ProjectManager":
        from rag.project.manager import ProjectManager
        return ProjectManager
    elif name == "MetadataManager":
        from rag.project.metadata import MetadataManager
        return MetadataManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ProjectManager", "MetadataManager"]
