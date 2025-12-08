"""
Document ingestion MCP tools.
"""

import os
from datetime import datetime
from pathlib import Path

from llama_index.core import Document, Settings

from rag.config import settings, logger, CODE_INDICATOR_FILES, CODE_INDICATOR_DIRS
from rag.storage.index import get_index_manager
from rag.project.manager import get_project_manager
from rag.project.metadata import get_metadata_manager
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.processor import DocumentProcessor
from rag.ingestion.chunker import DocumentChunker


def register_ingest_tools(mcp):
    """Register ingestion MCP tools."""
    
    loader = DocumentLoader()
    processor = DocumentProcessor()
    chunker = DocumentChunker()
    
    @mcp.tool()
    def index_documents(
        path: str,
        project: str = None,
        mode: str = "auto"
    ) -> dict:
        """
        Index documents or code from a directory.
        
        Args:
            path: Directory path to index
            project: Project/workspace name (creates if doesn't exist)
            mode: 'auto' (detect), 'code' (codebase), 'docs' (documents), or 'hybrid' (mixed)
        
        Examples:
            index_documents(path='/path/to/docs', project='my-docs')
            index_documents(path='/path/to/repo', project='backend', mode='code')
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {"error": f"Path not found: {path}"}
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            # Resolve project
            pm = get_project_manager()
            suggested_name = dir_path.name
            
            if not project:
                return {
                    "project_confirmation_required": True,
                    "suggested_project": suggested_name,
                    "message": f"Specify project name or use suggested: '{suggested_name}'"
                }
            
            if project not in pm.list_projects():
                pm.create_project(project)
                logger.info(f"Created new project: {project}")
            
            get_index_manager().switch_project(project)
            
            # Detect mode
            if mode == "auto":
                has_code_markers = any(
                    (dir_path / f).exists() for f in CODE_INDICATOR_FILES
                ) or any(
                    (dir_path / d).exists() for d in CODE_INDICATOR_DIRS
                )
                mode = "hybrid" if has_code_markers else "docs"
            
            # Load documents
            logger.info(f"Loading documents from {path} (mode: {mode})")
            documents, skip_stats = loader.load_directory(
                directory=dir_path
            )
            
            if not documents:
                return {"error": "No supported documents found"}
            
            # Process and chunk
            documents = [processor.inject_context(doc) for doc in documents]
            all_nodes = []
            
            for doc in documents:
                nodes = chunker.chunk_document(doc)
                all_nodes.extend(nodes)
            
            # Insert
            index_manager = get_index_manager()
            index_manager.insert_nodes(all_nodes, project, show_progress=True)
            index_manager.persist(project)
            
            # Update metadata
            pm.update_project_paths(project, [str(dir_path)])
            get_metadata_manager().record_index_activity(project, str(dir_path))
            
            return {
                "success": True,
                "project": project,
                "documents_processed": len(documents),
                "chunks_created": len(all_nodes),
                "mode": mode
            }
            
        except Exception as e:
            logger.error(f"Indexing error: {e}", exc_info=True)
            return {"error": str(e)}
    
    @mcp.tool()
    def add_text(text: str, metadata: dict = None, project: str = None) -> dict:
        """
        Add raw text to index.
        
        Args:
            text: Content to add
            metadata: Optional metadata dict
            project: Target project (uses current if not specified)
        """
        try:
            if not text or not text.strip():
                return {"error": "Text cannot be empty"}
            
            doc_metadata = processor.sanitize_metadata(metadata or {})
            doc_metadata['added_via'] = 'mcp_tool'
            doc_metadata['added_at'] = datetime.now().isoformat()
            
            document = Document(text=text, metadata=doc_metadata)
            document = processor.inject_context(document)
            
            index_manager = get_index_manager()
            index = index_manager.get_index(project)
            
            nodes = Settings.text_splitter.get_nodes_from_documents([document])
            index.insert_nodes(nodes)
            index_manager.persist(project)
            
            target_project = project or index_manager.current_project
            get_metadata_manager().record_index_activity(target_project, "inline_text")
            
            return {
                "success": True,
                "chunks_created": len(nodes),
                "text_length": len(text)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def inspect_directory(path: str) -> dict:
        """
        Analyze directory contents before indexing.
        
        Args:
            path: Directory path to inspect
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {"error": f"Path not found: {path}"}
            
            stats = {
                "code_files": 0,
                "doc_files": 0,
                "total_size_mb": 0,
                "has_git": (dir_path / ".git").exists(),
                "suggested_name": dir_path.name,
                "suggested_mode": "docs"
            }
            
            for file in dir_path.rglob("*"):
                if file.is_file():
                    ext = file.suffix.lower()
                    size_mb = file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += size_mb
                    
                    if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.go']:
                        stats["code_files"] += 1
                    elif ext in ['.md', '.txt', '.pdf', '.docx']:
                        stats["doc_files"] += 1
            
            if stats["code_files"] > stats["doc_files"]:
                stats["suggested_mode"] = "code"
            elif stats["code_files"] > 0:
                stats["suggested_mode"] = "hybrid"
            
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
