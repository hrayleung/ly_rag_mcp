"""
Admin and management MCP tools.
"""

from rag.config import settings, logger
from rag.storage.index import get_index_manager
from rag.storage.chroma import get_chroma_manager
from rag.project.manager import get_project_manager


def register_admin_tools(mcp):
    """Register admin MCP tools."""
    
    @mcp.tool()
    def manage_project(
        action: str,
        project: str = None,
        keywords: list[str] = None,
        description: str = None
    ) -> dict:
        """
        Manage projects (workspaces) for organizing different knowledge bases.
        
        Args:
            action: Operation to perform:
                - 'list': Show all projects with their metadata
                - 'create': Create new project
                - 'switch': Switch to different project
                - 'update': Update project keywords/description for better routing
                - 'analyze': Get routing suggestions for a query
            project: Project name (required for create/switch/update)
            keywords: Keywords for auto-routing (for update action)
                      Examples: ['python', 'api'], ['frontend', 'react'], ['backend', 'database']
            description: Human-readable description (for update action)
        
        Examples:
            # List all projects with metadata
            manage_project(action='list')
            
            # Create and configure a new project
            manage_project(action='create', project='backend-api')
            manage_project(action='update', project='backend-api', 
                          keywords=['python', 'fastapi', 'rest', 'database'],
                          description='Backend REST API documentation')
            
            # Test routing for a query
            manage_project(action='analyze', description='How to setup FastAPI endpoints?')
        
        Note: Good keywords improve automatic project routing. Include:
        - Technology names (python, react, aws)
        - Domain terms (api, frontend, database)
        - Project-specific terms (auth, payment, analytics)
        """
        try:
            pm = get_project_manager()
            
            if action == "list":
                project_info = pm.list_projects()
                current = get_index_manager().current_project
                
                # Use the details from list_projects
                return {
                    "projects": project_info.get("details", []),
                    "current": current,
                    "total": project_info.get("count", 0)
                }
            
            elif action == "create":
                if not project:
                    return {"error": "project name required"}
                pm.create_project(project)
                return {
                    "success": f"Created project: {project}",
                    "next_step": f"Add keywords: manage_project(action='update', project='{project}', keywords=['term1', 'term2'])"
                }
            
            elif action == "switch":
                if not project:
                    return {"error": "project name required"}
                get_index_manager().switch_project(project)
                return {"success": f"Switched to: {project}"}
            
            elif action == "update":
                if not project:
                    return {"error": "project name required"}
                if not keywords and not description:
                    return {"error": "Provide keywords and/or description"}
                
                pm.set_project_metadata(project, keywords or [], description)
                return {
                    "success": f"Updated metadata for: {project}",
                    "keywords": keywords,
                    "description": description,
                    "tip": "Good keywords improve auto-routing accuracy"
                }
            
            elif action == "analyze":
                if not description:
                    return {"error": "Provide a query in 'description' field to analyze routing"}
                
                result = pm.choose_project(description, max_candidates=3)
                if "error" in result:
                    return result
                
                return {
                    "query": description,
                    "recommendation": result.get("recommendation"),
                    "all_candidates": result.get("candidates", []),
                    "tip": "Top candidate will be auto-selected if you don't specify project in query_rag"
                }
            
            else:
                return {
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["list", "create", "switch", "update", "analyze"]
                }
                
        except Exception as e:
            logger.error(f"Project management error: {e}", exc_info=True)
            return {"error": str(e)}
    
    @mcp.tool()
    def get_stats(stat_type: str = "index") -> dict:
        """
        Get system statistics.
        
        Args:
            stat_type: 'index' (default) or 'cache'
        """
        try:
            if stat_type == "cache":
                stats = get_index_manager().stats
                return {
                    "index_loads": stats.index_loads,
                    "index_cache_hits": stats.index_cache_hits,
                    "cache_hit_rate": f"{stats.cache_hit_rate:.1%}"
                }
            
            # Default: index stats
            if not settings.storage_path.exists():
                return {"error": "Index not found"}
            
            return {
                "status": "ready",
                "documents": get_chroma_manager().get_collection_count(),
                "current_project": get_index_manager().current_project,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def list_documents(project: str = None, limit: int = 50) -> dict:
        """
        List indexed documents.
        
        Args:
            project: Project name (optional, uses current)
            limit: Max documents to return (default: 50)
        """
        try:
            index = get_index_manager().get_index(project)
            docstore = getattr(index, "docstore", None)
            if not docstore:
                return {"documents": [], "message": "No docstore available"}
            
            docs = getattr(docstore, "docs", {}) or {}
            doc_list = []
            
            for doc_id, doc in list(docs.items())[:limit]:
                metadata = getattr(doc, "metadata", {}) or {}
                doc_list.append({
                    "id": doc_id[:16],
                    "file": metadata.get("file_name", "unknown"),
                    "type": metadata.get("file_type", "unknown"),
                    "size": metadata.get("file_size", 0)
                })
            
            return {
                "documents": doc_list,
                "total": len(docs),
                "showing": len(doc_list)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def clear_index(project: str = None, confirm: bool = False) -> dict:
        """
        Clear the index (DESTRUCTIVE).
        
        Args:
            project: Project to clear (optional, uses current)
            confirm: Must be True to proceed
        """
        if not confirm:
            return {"error": "Set confirm=True to proceed"}
        
        try:
            target = project or get_index_manager().current_project
            get_chroma_manager().clear_collection(target)
            get_index_manager().reset()
            return {"success": f"Cleared index for: {target}"}
        except Exception as e:
            return {"error": str(e)}
