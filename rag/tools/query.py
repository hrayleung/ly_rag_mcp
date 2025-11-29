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
        use_hyde: bool = False
    ) -> str:
        """
        Retrieve documents from RAG.
        
        Args:
            question: Query string.
            similarity_top_k: Documents to return (default: 6).
            search_mode: 'semantic' (default), 'hybrid', or 'keyword'.
            use_rerank: Enable Cohere reranking (default: True).
            use_hyde: Enable HyDE for ambiguous queries (default: False).
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
                return "No relevant documents found in the knowledge base."
            
            # Format output
            mode_label = f" ({result.search_mode.value}"
            if result.used_hyde:
                mode_label += " + HyDE"
            if result.reranked:
                mode_label += " + Cohere Rerank"
            mode_label += ")"
            
            output = f"Found {result.total} relevant document(s){mode_label}:\n\n"
            
            for i, doc in enumerate(result.results, 1):
                output += f"--- Document {i} ---\n"
                output += f"{doc.text}\n"
                if doc.metadata:
                    output += f"\nMetadata: {doc.metadata}\n"
                if doc.score:
                    output += f"Relevance Score: {doc.score:.3f}\n"
                output += "\n"
            
            return output
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"Invalid input: {str(e)}"
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return f"Error retrieving documents: {str(e)}"
    
    @mcp.tool()
    @log_performance
    def query_rag_with_sources(
        question: str,
        similarity_top_k: int = settings.default_top_k,
        search_mode: str = "semantic",
        use_rerank: bool = True,
        use_hyde: bool = False
    ) -> dict:
        """
        Retrieve documents with metadata (structured).
        
        Args:
            question: Query string.
            similarity_top_k: Documents to return.
            search_mode: 'semantic', 'hybrid', 'keyword'.
            use_rerank: Enable reranking (default: True).
            use_hyde: Enable HyDE (default: False).
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
                return {
                    "sources": [],
                    "message": "No relevant documents found"
                }
            
            sources = [
                {
                    "text": r.text,
                    "text_preview": r.preview,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in result.results
            ]
            
            return {
                "sources": sources,
                "total_found": result.total,
                "search_mode": result.search_mode.value,
                "reranked": result.reranked,
                "used_hyde": result.used_hyde,
                "generated_query": result.generated_query
            }
            
        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            logger.error(f"Error querying RAG: {e}", exc_info=True)
            return {"error": f"Error querying RAG: {str(e)}"}
    
    @mcp.tool()
    @log_performance
    def iterative_search(
        question: str,
        initial_top_k: int = 3,
        detailed_top_k: int = 10,
        search_mode: str = "semantic",
        use_rerank: bool = True,
        use_hyde: bool = False
    ) -> dict:
        """
        Two-phase search: quick initial results + suggestions.
        
        Args:
            question: Query string.
            initial_top_k: Initial results (default: 3).
            detailed_top_k: Potential results for follow-up (default: 10).
            search_mode: 'semantic', 'hybrid', 'keyword'.
            use_rerank: Enable reranking.
            use_hyde: Enable HyDE.
        """
        try:
            engine = get_search_engine()
            result = engine.search(
                question=question,
                similarity_top_k=initial_top_k,
                search_mode=search_mode,
                use_rerank=use_rerank,
                use_hyde=use_hyde
            )
            
            if not result.results:
                return {
                    "initial_results": [],
                    "message": "No relevant documents found",
                    "suggestion": "Try rephrasing or different keywords"
                }
            
            initial_results = [
                {
                    "rank": i,
                    "text": r.text,
                    "preview": r.preview,
                    "score": r.score,
                    "metadata": r.metadata,
                    "length": len(r.text)
                }
                for i, r in enumerate(result.results, 1)
            ]
            
            # Analyze scores
            scores = [r["score"] for r in initial_results if r["score"]]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            suggestions = []
            if result.search_mode.value == "semantic" and avg_score < 0.7:
                suggestions.append("Low relevance. Try search_mode='hybrid' for technical terms.")
            if avg_score < 0.7:
                suggestions.append(f"Relevance moderate (avg: {avg_score:.2f}). Consider refining query.")
            suggestions.append(
                f"Found {len(initial_results)} initial. "
                f"Get more via query_rag(similarity_top_k={detailed_top_k})"
            )
            
            return {
                "initial_results": initial_results,
                "total_initial": len(initial_results),
                "search_mode": result.search_mode.value,
                "reranked": result.reranked,
                "used_hyde": result.used_hyde,
                "suggestions": suggestions,
                "next_steps": [
                    "If satisfied: Answer the question",
                    f"If need more: query_rag('{question}', similarity_top_k={detailed_top_k})",
                    "For exact terms: search_mode='keyword'"
                ]
            }
            
        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            logger.error(f"Error in iterative search: {e}", exc_info=True)
            return {"error": f"Error in iterative search: {str(e)}"}
