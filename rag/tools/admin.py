"""
Administration and management MCP tools.
"""

import shutil
from pathlib import Path

from rag.config import settings, logger
from rag.storage.index import get_index_manager
from rag.storage.chroma import get_chroma_manager
from rag.project.manager import get_project_manager
from rag.project.metadata import get_metadata_manager
from rag.retrieval.reranker import get_reranker_manager
from rag.retrieval.bm25 import get_bm25_manager


def register_admin_tools(mcp):
    """Register administration MCP tools."""
    
    @mcp.tool()
    def create_project(name: str) -> dict:
        """Create a new isolated project (workspace)."""
        return get_project_manager().create_project(name)
    
    @mcp.tool()
    def list_projects() -> dict:
        """List all available projects."""
        return get_project_manager().list_projects()
    
    @mcp.tool()
    def switch_project(name: str) -> dict:
        """Switch the active workspace."""
        return get_project_manager().switch_project(name)
    
    @mcp.tool()
    def set_project_metadata(
        project: str,
        display_name: str = None,
        description: str = None,
        keywords: list = None,
        default_paths: list = None
    ) -> dict:
        """
        Define metadata for better query routing.
        
        Args:
            project: Project name.
            display_name: Human-readable name.
            description: Project description.
            keywords: Search keywords.
            default_paths: Default source paths.
        """
        try:
            manager = get_project_manager()
            
            if not manager.project_exists(project):
                return {
                    "error": "project_not_found",
                    "available": manager.discover_projects()
                }
            
            metadata_mgr = get_metadata_manager()
            metadata = metadata_mgr.update(
                project=project,
                display_name=display_name,
                description=description,
                keywords=keywords,
                default_paths=default_paths
            )
            
            return {
                "success": True,
                "project": project,
                "metadata": metadata.to_dict()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def choose_project(question: str, max_candidates: int = 3) -> dict:
        """
        Score available projects for a query without executing retrieval.
        
        Args:
            question: User query to evaluate.
            max_candidates: Maximum candidates to return.
        """
        return get_project_manager().choose_project(question, max_candidates)
    
    @mcp.tool()
    def get_index_stats() -> dict:
        """Get RAG index statistics."""
        try:
            if not settings.storage_path.exists():
                return {"error": "Index not found. Run build_index.py first."}
            
            chroma_manager = get_chroma_manager()
            count = chroma_manager.get_collection_count()
            index_manager = get_index_manager()
            
            return {
                "status": "ready",
                "document_count": count,
                "current_project": index_manager.current_project,
                "storage_location": str(settings.storage_path),
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            return {"error": f"Error getting stats: {str(e)}"}
    
    @mcp.tool()
    def get_cache_stats() -> dict:
        """Get cache performance metrics."""
        index_stats = get_index_manager().stats
        chroma_stats = get_chroma_manager().stats
        reranker_stats = get_reranker_manager().stats
        bm25_stats = get_bm25_manager().stats
        
        # Combine stats
        combined = {
            "index": {
                "loads": index_stats.index_loads,
                "cache_hits": index_stats.index_cache_hits,
                "hit_rate": f"{index_stats.get_hit_rate('index'):.1f}%"
            },
            "chroma": {
                "loads": chroma_stats.chroma_loads,
                "cache_hits": chroma_stats.chroma_cache_hits,
                "hit_rate": f"{chroma_stats.get_hit_rate('chroma'):.1f}%"
            },
            "reranker": {
                "loads": reranker_stats.reranker_loads,
                "cache_hits": reranker_stats.reranker_cache_hits,
                "hit_rate": f"{reranker_stats.get_hit_rate('reranker'):.1f}%"
            },
            "bm25": {
                "builds": bm25_stats.bm25_builds,
                "cache_hits": bm25_stats.bm25_cache_hits,
                "hit_rate": f"{bm25_stats.get_hit_rate('bm25'):.1f}%"
            }
        }
        
        # Summary
        min_rate = min(
            index_stats.get_hit_rate('index'),
            chroma_stats.get_hit_rate('chroma')
        )
        combined["summary"] = (
            "Cache working well" if min_rate > 50 
            else "Cache warming up"
        )
        
        return combined
    
    @mcp.tool()
    def list_indexed_documents() -> dict:
        """List sample of indexed documents."""
        try:
            stats = get_index_stats()
            if "error" in stats:
                return stats
            
            count = stats.get("document_count", 0)
            if count == 0:
                return {
                    "success": True,
                    "document_count": 0,
                    "documents": [],
                    "message": "No documents in the index"
                }
            
            index_manager = get_index_manager()
            retriever = index_manager.get_retriever(similarity_top_k=10)
            
            from llama_index.core.schema import QueryBundle
            query = QueryBundle(query_str="文档 内容 代码")
            nodes = retriever.retrieve(query)
            
            docs = []
            for i, node in enumerate(nodes, 1):
                content = node.node.get_content()
                docs.append({
                    "index": i,
                    "node_id": node.node.node_id,
                    "text_preview": content[:150] + "..." if len(content) > 150 else content,
                    "metadata": node.node.metadata,
                    "text_length": len(content),
                    "relevance_score": float(node.score) if node.score else None
                })
            
            return {
                "success": True,
                "document_count": count,
                "documents_shown": len(docs),
                "documents": docs,
                "message": f"Showing {len(docs)} samples of {count} total"
            }
        except Exception as e:
            return {"error": f"Error listing documents: {str(e)}"}
    
    @mcp.tool()
    def clear_index() -> dict:
        """Clear all documents (DESTRUCTIVE)."""
        try:
            if settings.storage_path.exists():
                shutil.rmtree(settings.storage_path)
            
            # Reset managers
            get_index_manager().reset()
            get_chroma_manager().reset()
            get_bm25_manager().reset()
            get_metadata_manager().clear_cache()
            
            # Reinitialize
            get_index_manager().get_index()
            
            return {
                "success": True,
                "message": "Index cleared successfully"
            }
        except Exception as e:
            return {"error": f"Error clearing index: {str(e)}"}
    
    # MCP Resources
    @mcp.resource("rag://documents")
    def get_documents_resource() -> str:
        """MCP Resource: List indexed documents."""
        try:
            result = list_indexed_documents()
            if "error" in result:
                return f"Error: {result['error']}"
            
            output = f"# Indexed Documents ({result['document_count']} total)\n\n"
            for doc in result["documents"]:
                output += f"## Document: {doc['node_id'][:8]}...\n"
                output += f"**Preview:** {doc['text_preview']}\n"
                output += f"**Length:** {doc['text_length']} chars\n"
                if doc['metadata']:
                    output += f"**Metadata:** {doc['metadata']}\n"
                output += "\n---\n\n"
            
            return output
        except Exception as e:
            return f"Error: {str(e)}"
    
    @mcp.resource("rag://stats")
    def get_stats_resource() -> str:
        """MCP Resource: Get RAG statistics."""
        try:
            stats = get_index_stats()
            if "error" in stats:
                return f"Error: {stats['error']}"
            
            output = "# RAG Index Statistics\n\n"
            for key, value in stats.items():
                output += f"**{key}:** {value}\n"
            
            return output
        except Exception as e:
            return f"Error: {str(e)}"
