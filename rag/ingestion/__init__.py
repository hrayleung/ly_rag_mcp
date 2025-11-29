"""
Ingestion layer for the RAG system.

Handles document loading, processing, and chunking.
"""

# Lazy imports
def __getattr__(name):
    if name == "DocumentLoader":
        from rag.ingestion.loader import DocumentLoader
        return DocumentLoader
    elif name == "DocumentProcessor":
        from rag.ingestion.processor import DocumentProcessor
        return DocumentProcessor
    elif name == "get_splitter_for_file":
        from rag.ingestion.chunker import get_splitter_for_file
        return get_splitter_for_file
    elif name == "ChunkingStrategy":
        from rag.ingestion.chunker import ChunkingStrategy
        return ChunkingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DocumentLoader",
    "DocumentProcessor",
    "get_splitter_for_file",
    "ChunkingStrategy",
]
