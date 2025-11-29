"""
Retrieval layer for the RAG system.

Handles different retrieval strategies: semantic, hybrid, BM25.
"""

# Lazy imports
def __getattr__(name):
    if name == "SearchEngine":
        from rag.retrieval.search import SearchEngine
        return SearchEngine
    elif name == "RerankerManager":
        from rag.retrieval.reranker import RerankerManager
        return RerankerManager
    elif name == "BM25Manager":
        from rag.retrieval.bm25 import BM25Manager
        return BM25Manager
    elif name == "generate_hyde_query":
        from rag.retrieval.hyde import generate_hyde_query
        return generate_hyde_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SearchEngine",
    "RerankerManager", 
    "BM25Manager",
    "generate_hyde_query",
]
