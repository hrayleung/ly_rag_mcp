"""
Query and retrieval MCP tools.
"""

from functools import wraps
import time
from time import perf_counter

from rag.config import settings, logger
from rag.retrieval.search import get_search_engine
from rag.project.manager import get_project_manager
from rag.project.metadata import get_metadata_manager

# Minimum score threshold to consider results relevant
RELEVANCE_THRESHOLD = 0.5


def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        name = func.__name__
        logger.info(f"Starting {name}")

        try:
            result = func(*args, **kwargs)
            elapsed = perf_counter() - start
            logger.info(f"Completed {name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = perf_counter() - start
            logger.error(f"Failed {name} after {elapsed:.3f}s: {e}")
            raise

    return wrapper


def _search_project(engine, question: str, project: str, **kwargs):
    """Execute search on a specific project and return results with relevance check."""
    result = engine.search(
        question=question,
        project=project,
        **kwargs
    )
    
    # Check if results are relevant (score above threshold)
    is_relevant = False
    if result.results:
        scores = []
        for r in result.results:
            try:
                if r.score is not None:
                    scores.append(float(r.score))
            except Exception:
                continue
        top_score = max(scores) if scores else 0.0
        is_relevant = top_score >= RELEVANCE_THRESHOLD

    return result, is_relevant


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
        based on your question and project metadata. If results are not relevant, it will
        try other projects. Successful queries update project keywords for better future routing.
        """
        try:
            pm = get_project_manager()
            mm = get_metadata_manager()
            engine = get_search_engine()
            
            search_kwargs = {
                "similarity_top_k": similarity_top_k,
                "search_mode": search_mode,
                "use_rerank": use_rerank,
                "use_hyde": use_hyde
            }
            
            # If project specified, search only that project
            if project:
                result, is_relevant = _search_project(engine, question, project, **search_kwargs)
                if is_relevant:
                    mm.learn_from_query(project, question, success=True)
                return _format_result(result, project, False, return_metadata)
            
            # Auto-routing: get ranked project candidates
            projects = pm.discover_projects()
            if not projects:
                msg = "No projects available"
                return {"error": msg} if return_metadata else msg
            
            if len(projects) == 1:
                # Only one project, use it directly
                result, is_relevant = _search_project(engine, question, projects[0], **search_kwargs)
                if is_relevant:
                    mm.learn_from_query(projects[0], question, success=True)
                return _format_result(result, projects[0], False, return_metadata)
            
            # Multiple projects - use smart routing with fallback
            routing = pm.choose_project(question, max_candidates=len(projects))
            candidates = routing.get("candidates", [])
            
            if not candidates:
                msg = "No matching projects found"
                return {"error": msg} if return_metadata else msg
            
            tried_projects = []
            first_result = None
            first_project = None
            for idx, candidate in enumerate(candidates):
                proj_name = candidate["project"]
                tried_projects.append(proj_name)

                result, is_relevant = _search_project(engine, question, proj_name, **search_kwargs)

                if idx == 0:
                    first_result = result
                    first_project = proj_name

                if is_relevant:
                    # Found relevant results - learn and return
                    mm.learn_from_query(proj_name, question, success=True)
                    logger.info(f"Found relevant results in project: {proj_name}")
                    return _format_result(result, proj_name, True, return_metadata, tried_projects)

                logger.info(f"Results from {proj_name} not relevant (score < {RELEVANCE_THRESHOLD}), trying next...")

            # No relevant results in any project - return best attempt
            if first_result is None:
                msg = "No matching projects found"
                return {"error": msg} if return_metadata else msg
            return _format_result(first_result, first_project, True, return_metadata, tried_projects,
                                  note="No highly relevant results found in any project")
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {"error": str(e)} if return_metadata else f"Error: {str(e)}"


def _format_result(result, project: str, auto_routed: bool, return_metadata: bool, 
                   tried_projects: list = None, note: str = None) -> str | dict:
    """Format search results for output."""
    if not result.results:
        msg = f"No relevant documents found [project: {project}]"
        return {"error": msg, "project": project} if return_metadata else msg
    
    if return_metadata:
        output = {
            "sources": [
                {
                    "text": r.text,
                    "score": (float(r.score) if r.score is not None else None),
                    "metadata": r.metadata,
                }
                for r in result.results
            ],
            "total": result.total,
            "search_mode": result.search_mode.value,
            "reranked": result.reranked,
            "used_hyde": result.used_hyde,
            "project": project,
            "auto_routed": auto_routed,
        }
        if tried_projects:
            output["tried_projects"] = tried_projects
        if note:
            output["note"] = note
        return output
    
    # Format as text
    routing_info = f" [project: {project}]"
    if tried_projects and len(tried_projects) > 1:
        routing_info = f" [tried: {' → '.join(tried_projects)}]"
    
    mode_info = f" ({result.search_mode.value}"
    if result.used_hyde:
        mode_info += " + HyDE"
    if result.reranked:
        mode_info += " + reranked"
    mode_info += ")"
    
    output = f"Found {result.total} documents{mode_info}{routing_info}:\n"
    if note:
        output += f"Note: {note}\n"
    output += "\n"
    
    for i, doc in enumerate(result.results, 1):
        if doc.score is None:
            score_str = "n/a"
        else:
            try:
                score_str = f"{float(doc.score):.3f}"
            except Exception:
                score_str = "n/a"
        output += f"--- Document {i} (score: {score_str}) ---\n{doc.text}\n\n"

    return output
