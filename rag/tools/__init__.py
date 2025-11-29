"""
MCP tool definitions for the RAG system.

Tools are organized by category:
- query: Search and retrieval tools
- ingest: Document ingestion tools  
- admin: Administration and management tools
"""

from rag.tools.query import register_query_tools
from rag.tools.ingest import register_ingest_tools
from rag.tools.admin import register_admin_tools


def register_all_tools(mcp):
    """Register all MCP tools."""
    register_query_tools(mcp)
    register_ingest_tools(mcp)
    register_admin_tools(mcp)


__all__ = [
    "register_all_tools",
    "register_query_tools",
    "register_ingest_tools",
    "register_admin_tools",
]
