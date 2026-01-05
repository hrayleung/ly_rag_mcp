#!/usr/bin/env python3
"""
MCP Server for querying the local RAG database.

This is a clean entry point that initializes the FastMCP server
and registers all tools from the modular rag package.

Features:
- Multi-project isolation
- Hybrid search (semantic + BM25)
- Cohere reranking
- HyDE query augmentation
- Incremental indexing
"""

import nest_asyncio
from mcp.server.fastmcp import FastMCP

# Allow nested event loops (needed for some readers)
nest_asyncio.apply()

# Validate API keys at startup
from rag.config import validate_required_keys, logger
try:
    validate_required_keys()
    logger.info("API key validation passed")
except ValueError as e:
    logger.error(f"API key validation failed: {e}")
    raise

# Initialize FastMCP server
mcp = FastMCP("LlamaIndex RAG")

# Register all tools from the rag package
from rag.tools import register_all_tools
register_all_tools(mcp)


# Export commonly needed items for backward compatibility
def get_index():
    """Get the current index (backward compatibility)."""
    from rag.storage.index import get_index_manager
    return get_index_manager().get_index()


def get_reranker():
    """Get the reranker (backward compatibility)."""
    from rag.retrieval.reranker import get_reranker_manager
    return get_reranker_manager().get_reranker()


def get_index_stats():
    """Get index statistics (backward compatibility)."""
    from rag.storage.index import get_index_manager
    from rag.storage.chroma import get_chroma_manager
    from rag.config import settings
    
    if not settings.storage_path.exists():
        return {"error": "Index not found"}
    
    return {
        "status": "ready",
        "document_count": get_chroma_manager().get_collection_count(),
        "current_project": get_index_manager().current_project,
        "storage_location": str(settings.storage_path),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model
    }


def query_rag(question: str, similarity_top_k: int = 6, **kwargs) -> str:
    """Query the RAG system (backward compatibility)."""
    from rag.retrieval.search import get_search_engine
    
    result = get_search_engine().search(
        question=question,
        similarity_top_k=similarity_top_k,
        **kwargs
    )
    
    if not result.results:
        return "No relevant documents found."
    
    output = f"Found {result.total} relevant document(s):\n\n"
    for i, doc in enumerate(result.results, 1):
        output += f"--- Document {i} ---\n{doc.text}\n"
        if doc.score is not None:
            output += f"Score: {doc.score:.3f}\n"
        output += "\n"
    
    return output


def iterative_search(question: str, initial_top_k: int = 3, **kwargs) -> dict:
    """Iterative search (backward compatibility)."""
    from rag.retrieval.search import get_search_engine
    
    result = get_search_engine().search(
        question=question,
        similarity_top_k=initial_top_k,
        **kwargs
    )
    
    return {
        "initial_results": [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in result.results
        ],
        "total_initial": result.total,
        "search_mode": result.search_mode.value
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
