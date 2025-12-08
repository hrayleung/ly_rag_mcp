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
        Manage projects (workspaces).
        
        Args:
            action: 'list', 'create', 'switch', 'set_metadata', or 'choose'
            project: Project name (required for create/switch/set_metadata)
            keywords: Keywords for routing (for set_metadata)
            description: Project description (for set_metadata)
        
        Examples:
            manage_project(action='list')
            manage_project(action='create', project='backend')
            manage_project(action='switch', project='frontend')
            manage_project(action='set_metadata', project='api', keywords=['rest','graphql'], description='API docs')
        """
        try:
            pm = get_project_manager()
            
            if action == "list":
                projects = pm.list_projects()
                return {
                    "projects": projects,
                    "current": get_index_manager().current_project
                }
            
            elif action == "create":
                if not project:
                    return {"error": "project name required"}
                pm.create_project(project)
                return {"success": f"Created project: {project}"}
            
            elif action == "switch":
                if not project:
                    return {"error": "project name required"}
                get_index_manager().switch_project(project)
                return {"success": f"Switched to: {project}"}
            
            elif action == "set_metadata":
                if not project:
                    return {"error": "project name required"}
                pm.set_project_metadata(project, keywords or [], description)
                return {"success": f"Updated metadata for: {project}"}
            
            elif action == "choose":
                if not description:
                    return {"error": "description/question required for choosing"}
                result = pm.choose_project(description)
                return result
            
            else:
                return {"error": f"Unknown action: {action}. Use: list, create, switch, set_metadata, choose"}
                
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
