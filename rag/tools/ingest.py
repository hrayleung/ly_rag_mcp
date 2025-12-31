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

try:
    from firecrawl import Firecrawl
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


def register_ingest_tools(mcp):
    """Register ingestion MCP tools."""

    @mcp.tool()
    def index_documents(
        path: str,
        project: str = None
    ) -> dict:
        """
        Index documents or code from a directory.

        Args:
            path: Directory path to index
            project: Project/workspace name (creates if doesn't exist)

        Examples:
            index_documents(path='/path/to/docs', project='my-docs')
            index_documents(path='/path/to/repo', project='backend')
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {"error": f"Path not found: {path}"}
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            pm = get_project_manager()
            existing_projects = pm.discover_projects()
            suggested_name = dir_path.name
            
            # Always require explicit project selection
            if not project:
                # Extract preview keywords to help user decide
                preview_keywords = get_metadata_manager().extract_keywords_from_directory(dir_path, max_keywords=8)
                
                return {
                    "action_required": "select_project",
                    "message": "Please specify which project to index into, or create a new one.",
                    "path": str(dir_path),
                    "suggested_new_project": suggested_name,
                    "detected_keywords": preview_keywords,
                    "existing_projects": existing_projects,
                    "options": [
                        f"Create new project: index_documents(path='{path}', project='{suggested_name}')",
                        "Use existing project: index_documents(path='...', project='<existing_name>')"
                    ]
                }
            
            # Create new project if needed
            if project not in existing_projects:
                create_result = pm.create_project(project)
                if create_result.get("error"):
                    return create_result
                logger.info(f"Created new project: {project}")
                # Auto-generate initial keywords from directory
                keywords = get_metadata_manager().extract_keywords_from_directory(dir_path)
                if keywords:
                    pm.set_project_metadata(project, keywords=keywords)
                    logger.info(f"Auto-generated keywords for {project}: {keywords}")

            get_index_manager().switch_project(project)

            # Detect mode (removed; always infer based on content during processing)
            has_code_markers = any(
                (dir_path / f).exists() for f in CODE_INDICATOR_FILES
            ) or any(
                (dir_path / d).exists() for d in CODE_INDICATOR_DIRS
            )
            detected_mode = "hybrid" if has_code_markers else "docs"

            # Create per-call instances for thread safety
            loader = DocumentLoader()
            processor = DocumentProcessor()
            chunker = DocumentChunker()

            # Load documents
            logger.info(f"Loading documents from {path} (mode: {detected_mode})")
            documents, skip_stats = loader.load_directory(
                directory=dir_path
            )
            
            if not documents:
                return {"error": "No supported documents found"}
            
            # Process: sanitize metadata and inject context
            for doc in documents:
                doc.metadata = processor.sanitize_metadata(doc.metadata)
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
                "mode": detected_mode
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

            # Create per-call processor instance for thread safety
            processor = DocumentProcessor()
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
            
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)

            return stats
            
        except Exception as e:
            return {"error": str(e)}

    
    @mcp.tool()
    def crawl_website(url: str, max_pages: int = 10, project: str = None) -> dict:
        """
        Crawl and index a website using Firecrawl.

        Args:
            url: Website URL to crawl
            max_pages: Maximum pages to crawl (default: 10)
            project: Target project (uses current if not specified)
        """
        if not FIRECRAWL_AVAILABLE:
            return {"error": "Firecrawl not installed. Run: pip install firecrawl-py"}

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return {"error": "FIRECRAWL_API_KEY not set"}

        if max_pages <= 0:
            return {"error": "max_pages must be positive"}

        try:
            from firecrawl import Firecrawl
            from firecrawl.types import ScrapeOptions

            # Create per-call processor and chunker for thread safety
            processor = DocumentProcessor()
            chunker = DocumentChunker()

            app = Firecrawl(api_key=api_key)
            result = app.crawl(
                url,
                limit=max_pages,
                scrape_options=ScrapeOptions(formats=['markdown']),
                timeout=30  # 30 second timeout
            )

            # Validate result.data exists before accessing
            if not result or not hasattr(result, 'data') or not result.data:
                return {"error": "No content crawled"}

            documents = []
            for page in result.data:
                doc = Document(
                    text=page.markdown if hasattr(page, 'markdown') else str(page),
                    metadata={
                        'url': page.url if hasattr(page, 'url') else url,
                        'title': getattr(page.metadata, 'title', '') if hasattr(page, 'metadata') else '',
                        'source': 'firecrawl',
                        'crawled_at': datetime.now().isoformat()
                    }
                )
                documents.append(processor.inject_context(doc))

            all_nodes = []
            for doc in documents:
                all_nodes.extend(chunker.chunk_document(doc))
            
            index_manager = get_index_manager()
            index_manager.insert_nodes(all_nodes, project, show_progress=True)
            index_manager.persist(project)
            
            target_project = project or index_manager.current_project
            get_metadata_manager().record_index_activity(target_project, url)
            
            return {
                "success": True,
                "pages_crawled": len(documents),
                "chunks_created": len(all_nodes),
                "url": url
            }
        except Exception as e:
            logger.error(f"Crawl error: {e}", exc_info=True)
            return {"error": str(e)}
