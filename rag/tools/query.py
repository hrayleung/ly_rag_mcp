"""
Query and retrieval MCP tools.
"""

from functools import wraps
import time

from rag.config import settings, logger
from rag.retrieval.search import get_search_engine


def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        name = func.__name__
        logger.info(f"Starting {name}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"Completed {name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Failed {name} after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


def register_query_tools(mcp):
    """Register query-related MCP tools."""
    
    @mcp.tool()
    @log_performance
    def query_rag(
        question: str,
        similarity_top_k: int = settings.default_top_k,
        search_mode: str = "semantic",
        use_rerank: bool = True,
        use_hyde: bool = False,
        return_metadata: bool = False
    ) -> str | dict:
        """
        Search the knowledge base.
        
        Args:
            question: Query string
            similarity_top_k: Number of results (1-50, default: 6)
            search_mode: 'semantic' (default), 'hybrid' (code/technical), or 'keyword' (exact match)
            use_rerank: Enable Cohere reranking for better accuracy (default: True)
            use_hyde: Enable HyDE for ambiguous queries (default: False)
            return_metadata: Return structured data with sources (default: False)
        """
        try:
            engine = get_search_engine()
            result = engine.search(
                question=question,
                similarity_top_k=similarity_top_k,
                search_mode=search_mode,
                use_rerank=use_rerank,
                use_hyde=use_hyde
            )
            
            if not result.results:
                return {"error": "No relevant documents found"} if return_metadata else "No relevant documents found."
            
            # Return structured data
            if return_metadata:
                return {
                    "sources": [
                        {
                            "text": r.text,
                            "score": r.score,
                            "metadata": r.metadata,
                        }
                        for r in result.results
                    ],
                    "total": result.total,
                    "search_mode": result.search_mode.value,
                    "reranked": result.reranked,
                    "used_hyde": result.used_hyde
                }
            
            # Return formatted text
            mode_info = f" ({result.search_mode.value}"
            if result.used_hyde:
                mode_info += " + HyDE"
            if result.reranked:
                mode_info += " + reranked"
            mode_info += ")"
            
            output = f"Found {result.total} documents{mode_info}:\n\n"
            for i, doc in enumerate(result.results, 1):
                output += f"--- Document {i} (score: {doc.score:.3f}) ---\n{doc.text}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {"error": str(e)} if return_metadata else f"Error: {str(e)}"
