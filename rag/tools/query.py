"""
Query and retrieval MCP tools.
"""

from functools import wraps
import time

from rag.config import settings, logger
from rag.retrieval.search import get_search_engine
from rag.project.manager import get_project_manager


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
        return_metadata: bool = False,
        project: str = None
    ) -> str | dict:
        """
        Search the knowledge base with automatic project routing.
        
        Args:
            question: Query string
            similarity_top_k: Number of results (1-50, default: 6)
            search_mode: 'semantic' (default), 'hybrid' (code/technical), or 'keyword' (exact match)
            use_rerank: Enable Cohere reranking for better accuracy (default: True)
            use_hyde: Enable HyDE for ambiguous queries (default: False)
            return_metadata: Return structured data with sources (default: False)
            project: Specific project to search (optional, auto-routes if not specified)
        
        Note: If project is not specified, the system automatically selects the best project
        based on your question and project metadata (keywords, descriptions).
        """
        try:
            # Auto-route to best project if not specified
            selected_project = project
            routing_info = ""
            
            if not project:
                pm = get_project_manager()
                projects = pm.discover_projects()
                
                if len(projects) > 1:
                    # Multiple projects - use smart routing
                    routing = pm.choose_project(question, max_candidates=1)
                    if not routing.get("error") and routing.get("recommendation"):
                        rec = routing["recommendation"]
                        # Handle both dict and string formats
                        if isinstance(rec, dict):
                            selected_project = rec.get("project")
                            score = rec.get("score", 0)
                        else:
                            selected_project = rec
                            score = 0
                        
                        if selected_project:
                            routing_info = f" [auto-routed to '{selected_project}' (confidence: {score:.1f})]"
                            logger.info(f"Auto-routed query to project: {selected_project} (score: {score})")
                elif len(projects) == 1:
                    selected_project = projects[0]
                    routing_info = f" [project: {selected_project}]"
            else:
                routing_info = f" [project: {project}]"
            
            # Execute search
            engine = get_search_engine()
            result = engine.search(
                question=question,
                similarity_top_k=similarity_top_k,
                search_mode=search_mode,
                use_rerank=use_rerank,
                use_hyde=use_hyde,
                project=selected_project
            )
            
            if not result.results:
                msg = f"No relevant documents found{routing_info}"
                return {"error": msg, "project": selected_project} if return_metadata else msg
            
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
                    "used_hyde": result.used_hyde,
                    "project": selected_project,
                    "auto_routed": not project
                }
            
            # Return formatted text
            mode_info = f" ({result.search_mode.value}"
            if result.used_hyde:
                mode_info += " + HyDE"
            if result.reranked:
                mode_info += " + reranked"
            mode_info += ")"
            
            output = f"Found {result.total} documents{mode_info}{routing_info}:\n\n"
            for i, doc in enumerate(result.results, 1):
                output += f"--- Document {i} (score: {doc.score:.3f}) ---\n{doc.text}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {"error": str(e)} if return_metadata else f"Error: {str(e)}"
