"""
Document ingestion MCP tools.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document, Settings

from rag.config import (
    settings, logger,
    SUPPORTED_CODE_EXTS, SUPPORTED_DOC_EXTS,
    CODE_INDICATOR_FILES, CODE_INDICATOR_DIRS,
    DEFAULT_EXCLUDES, LANGUAGE_TO_EXTENSIONS
)
from rag.storage.index import get_index_manager
from rag.project.manager import get_project_manager
from rag.project.metadata import get_metadata_manager
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.processor import DocumentProcessor
from rag.ingestion.chunker import DocumentChunker, get_splitter_for_file


def register_ingest_tools(mcp):
    """Register ingestion-related MCP tools."""
    
    loader = DocumentLoader()
    processor = DocumentProcessor()
    chunker = DocumentChunker()
    
    @mcp.tool()
    def add_document_from_text(text: str, metadata: dict = None) -> dict:
        """
        Add raw text to index.
        
        Args:
            text: Content to add.
            metadata: Optional dict.
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
            index = index_manager.get_index()
            
            nodes = Settings.text_splitter.get_nodes_from_documents([document])
            index.insert_nodes(nodes)
            index_manager.persist()
            
            project = index_manager.current_project
            get_metadata_manager().record_index_activity(project, "inline_text")
            
            return {
                "success": True,
                "message": f"Document added ({len(nodes)} chunks)",
                "chunks_created": len(nodes),
                "text_length": len(text)
            }
        except Exception as e:
            return {"error": f"Error adding document: {str(e)}"}
    
    @mcp.tool()
    def add_documents_from_directory(
        directory_path: str, 
        project: Optional[str] = None
    ) -> dict:
        """
        Index all docs in directory (recursive).
        
        Args:
            directory_path: Absolute path.
            project: Target project/workspace.
        """
        try:
            project_manager = get_project_manager()
            error, resolved_project = project_manager.require_project(
                "add_documents_from_directory",
                project,
                suggested=Path(directory_path).name if directory_path else None
            )
            if error:
                return error
            
            dir_path = Path(directory_path)
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory_path}"}
            if not dir_path.is_dir():
                return {"error": f"Path is not a directory: {directory_path}"}
            
            # Load documents
            documents, skip_stats = loader.load_directory(
                directory=dir_path,
                extra_metadata={'ingested_via': 'mcp_tool'}
            )
            
            if not documents:
                return {
                    "error": "No documents found",
                    "skipped_unsupported": len(skip_stats["unsupported"]),
                    "skipped_oversize": len(skip_stats["oversize"])
                }
            
            # Process and chunk
            processed = processor.process_documents(documents, dir_path)
            nodes = chunker.chunk_documents(processed)
            
            # Index
            index_manager = get_index_manager()
            index = index_manager.get_index(resolved_project)
            index.insert_nodes(nodes)
            index_manager.persist(resolved_project)
            
            get_metadata_manager().record_index_activity(resolved_project, str(dir_path))
            
            return {
                "success": True,
                "message": f"Ingested {len(documents)} documents ({len(nodes)} chunks)",
                "document_count": len(documents),
                "chunks_created": len(nodes),
                "directory": str(dir_path),
                "skipped_unsupported": len(skip_stats["unsupported"]),
                "skipped_oversize": len(skip_stats["oversize"])
            }
        except Exception as e:
            return {"error": f"Error ingesting documents: {str(e)}"}
    
    @mcp.tool()
    def index_local_codebase(
        directory_path: str,
        language_filter: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
        project: Optional[str] = None
    ) -> dict:
        """
        Index local code with smart filtering.
        
        Args:
            directory_path: Codebase root.
            language_filter: Languages (e.g., ['python', '.ts']).
            exclude_patterns: Glob patterns to ignore.
            include_hidden: Include dotfiles (default: False).
            project: Target project/workspace.
        """
        try:
            dir_path = Path(directory_path).resolve()
            
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory_path}"}
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {directory_path}"}
            
            project_manager = get_project_manager()
            error, resolved_project = project_manager.require_project(
                "index_local_codebase",
                project,
                suggested=dir_path.name
            )
            if error:
                return error
            
            # Build extension filter
            allowed_exts = set()
            if language_filter:
                for lang in language_filter:
                    if lang.lower() in LANGUAGE_TO_EXTENSIONS:
                        allowed_exts.update(LANGUAGE_TO_EXTENSIONS[lang.lower()])
                    elif lang.startswith('.'):
                        allowed_exts.add(lang.lower())
                    else:
                        allowed_exts.add(f".{lang.lower()}")
            else:
                allowed_exts = SUPPORTED_CODE_EXTS | {'.md', '.txt', '.json', '.yaml', '.yml'}
            
            # Build excludes
            all_excludes = set(DEFAULT_EXCLUDES)
            if exclude_patterns:
                all_excludes.update(exclude_patterns)
            
            # Scan and load
            files, skip_stats = loader.scan_directory(
                directory=dir_path,
                exclude_patterns=all_excludes,
                extension_filter=allowed_exts,
                include_hidden=include_hidden
            )
            
            if not files:
                return {
                    "error": "No files found matching criteria",
                    "filters": {
                        "languages": language_filter or "all code files",
                        "excluded": sorted(all_excludes)
                    }
                }
            
            documents = loader.load_files(
                files=files,
                source_directory=dir_path,
                extra_metadata={
                    'source': 'local_codebase',
                    'indexed_via': 'codebase_indexer',
                    'content_type': 'code'
                }
            )
            
            if not documents:
                return {"error": "Failed to load documents"}
            
            # Add language info
            for doc in documents:
                file_ext = Path(doc.metadata.get('file_path', '')).suffix.lower()
                for lang, exts in LANGUAGE_TO_EXTENSIONS.items():
                    if file_ext in exts:
                        doc.metadata['language'] = lang
                        break
            
            # Process and chunk
            processed = processor.process_documents(documents, dir_path)
            nodes = []
            for doc in processed:
                file_path = doc.metadata.get('file_path', '')
                splitter = get_splitter_for_file(file_path)
                nodes.extend(splitter.get_nodes_from_documents([doc]))
            
            # Index
            index_manager = get_index_manager()
            index = index_manager.get_index(resolved_project)
            index.insert_nodes(nodes, show_progress=True)
            index_manager.persist(resolved_project)
            
            get_metadata_manager().record_index_activity(resolved_project, str(dir_path))
            
            # Stats
            lang_stats = {}
            for doc in documents:
                lang = doc.metadata.get('language', 'unknown')
                lang_stats[lang] = lang_stats.get(lang, 0) + 1
            
            return {
                "success": True,
                "message": f"Indexed codebase at {dir_path}",
                "codebase_root": str(dir_path),
                "files_indexed": len(documents),
                "files_skipped": len(skip_stats["excluded"]) + len(skip_stats["unsupported"]),
                "chunks_created": len(nodes),
                "language_breakdown": lang_stats
            }
        except Exception as e:
            logger.error(f"Error indexing codebase: {e}", exc_info=True)
            return {"error": f"Error indexing codebase: {str(e)}"}
    
    @mcp.tool()
    def index_hybrid_folder(
        path: str,
        exclude_patterns: Optional[List[str]] = None,
        project: Optional[str] = None
    ) -> dict:
        """
        Index folder with mixed content (Code + Documents).
        
        Args:
            path: Directory path.
            exclude_patterns: Additional patterns to ignore.
            project: Target project/workspace.
        """
        try:
            dir_path = Path(path).resolve()
            
            if not dir_path.exists():
                return {"error": f"Path not found: {path}"}
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            project_manager = get_project_manager()
            error, resolved_project = project_manager.require_project(
                "index_hybrid_folder",
                project,
                suggested=dir_path.name
            )
            if error:
                return error
            
            # Mixed extensions
            valid_exts = SUPPORTED_CODE_EXTS | SUPPORTED_DOC_EXTS
            all_excludes = set(DEFAULT_EXCLUDES)
            if exclude_patterns:
                all_excludes.update(exclude_patterns)
            
            files, skip_stats = loader.scan_directory(
                directory=dir_path,
                exclude_patterns=all_excludes,
                extension_filter=valid_exts
            )
            
            if not files:
                return {"error": "No valid files found"}
            
            documents = loader.load_files(
                files=files,
                source_directory=dir_path,
                extra_metadata={'source': 'hybrid_folder'}
            )
            
            processed = processor.process_documents(documents, dir_path)
            nodes = chunker.chunk_documents(processed)
            
            index_manager = get_index_manager()
            index = index_manager.get_index(resolved_project)
            index.insert_nodes(nodes, show_progress=True)
            index_manager.persist(resolved_project)
            
            get_metadata_manager().record_index_activity(resolved_project, str(dir_path))
            
            return {
                "success": True,
                "message": f"Indexed {len(documents)} mixed files ({len(nodes)} chunks)",
                "files_indexed": len(documents),
                "chunks_created": len(nodes),
                "root": str(dir_path)
            }
        except Exception as e:
            logger.error(f"Error in hybrid indexing: {e}", exc_info=True)
            return {"error": f"Hybrid indexing failed: {str(e)}"}
    
    @mcp.tool()
    def inspect_directory(path: str) -> dict:
        """
        Analyze folder to recommend indexing method.
        
        Args:
            path: Absolute path to directory.
        """
        try:
            dir_path = Path(path).resolve()
            
            if not dir_path.exists():
                return {"error": f"Path not found: {path}"}
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            stats = {
                "total_files": 0,
                "code_files": 0,
                "doc_files": 0,
                "extensions": {},
                "markers": []
            }
            
            # Quick scan
            file_limit = 1000
            scanned = 0
            
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                
                if root_path == dir_path:
                    for d in dirs:
                        if d in CODE_INDICATOR_DIRS:
                            stats['markers'].append(d + "/")
                    for f in files:
                        if f in CODE_INDICATOR_FILES:
                            stats['markers'].append(f)
                
                for f in files:
                    if scanned >= file_limit:
                        break
                    
                    ext = Path(f).suffix.lower()
                    stats["total_files"] += 1
                    stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
                    
                    if ext in SUPPORTED_CODE_EXTS:
                        stats["code_files"] += 1
                    elif ext in SUPPORTED_DOC_EXTS:
                        stats["doc_files"] += 1
                    
                    scanned += 1
                
                if scanned >= file_limit:
                    break
            
            # Decision
            is_codebase = bool(stats['markers'])
            code_ratio = stats['code_files'] / stats['total_files'] if stats['total_files'] > 0 else 0
            doc_ratio = stats['doc_files'] / stats['total_files'] if stats['total_files'] > 0 else 0
            
            if stats['total_files'] == 0:
                recommendation = "none"
                reason = "Directory is empty"
            elif is_codebase or code_ratio > 0.5:
                recommendation = "index_local_codebase"
                reason = f"Code ratio {code_ratio:.1%}, markers: {', '.join(stats['markers'][:3])}"
            elif doc_ratio > 0.5:
                recommendation = "add_documents_from_directory"
                reason = f"Document ratio {doc_ratio:.1%}"
            else:
                recommendation = "index_hybrid_folder"
                reason = "Mixed content"
            
            project_manager = get_project_manager()
            
            return {
                "path": str(dir_path),
                "suggested_project_name": dir_path.name,
                "current_active_project": project_manager.current_project,
                "stats": {
                    "total_files": stats['total_files'] if scanned < file_limit else f"{file_limit}+",
                    "code_files": stats['code_files'],
                    "doc_files": stats['doc_files'],
                    "top_extensions": sorted(stats['extensions'].items(), key=lambda x: x[1], reverse=True)[:5]
                },
                "recommendation": recommendation,
                "reason": reason
            }
        except Exception as e:
            return {"error": f"Error inspecting directory: {str(e)}"}
    
    @mcp.tool()
    def crawl_website(url: str, max_depth: int = 1, max_pages: int = 10) -> dict:
        """
        Crawl website to RAG index (requires FIRECRAWL_API_KEY).
        
        Args:
            url: Starting URL.
            max_depth: 0=single page, 1=direct links, 2+=deep.
            max_pages: Limit pages.
        """
        try:
            from firecrawl import FirecrawlApp
            from rag.config import get_firecrawl_key
            
            api_key = get_firecrawl_key()
            if not api_key:
                return {"error": "FIRECRAWL_API_KEY not found"}
            
            if not url.startswith("http"):
                return {"error": "Invalid URL. Must start with http:// or https://"}
            
            app = FirecrawlApp(api_key=api_key)
            
            crawl_result = app.crawl(
                url,
                limit=max_pages,
                max_discovery_depth=max_depth,
                scrape_options={'formats': ['markdown']}
            )
            
            # Extract pages
            pages = []
            if hasattr(crawl_result, 'data'):
                pages = crawl_result.data
            elif isinstance(crawl_result, dict) and 'data' in crawl_result:
                pages = crawl_result['data']
            elif isinstance(crawl_result, list):
                pages = crawl_result
            
            if not pages:
                return {"error": "No pages found"}
            
            # Convert to documents
            documents = []
            for page in pages:
                if isinstance(page, dict):
                    text = page.get('markdown') or page.get('text')
                    source = page.get('source', url)
                else:
                    text = getattr(page, 'markdown', None) or getattr(page, 'text', None)
                    source = getattr(page, 'source', url)
                
                if not text:
                    continue
                
                doc = Document(
                    text=text,
                    metadata=processor.sanitize_metadata({
                        'source': source,
                        'crawled_at': datetime.now().isoformat(),
                        'crawled_via': 'firecrawl'
                    })
                )
                documents.append(doc)
            
            if not documents:
                return {"error": "No valid content extracted"}
            
            # Process and index
            processed = processor.process_documents(documents)
            nodes = chunker.chunk_documents(processed)
            
            index_manager = get_index_manager()
            index = index_manager.get_index()
            index.insert_nodes(nodes)
            index_manager.persist()
            
            get_metadata_manager().record_index_activity(
                index_manager.current_project, url
            )
            
            return {
                "success": True,
                "message": f"Crawled and indexed {len(documents)} pages from {url}",
                "pages_crawled": len(documents),
                "chunks_created": len(nodes)
            }
            
        except ImportError:
            return {"error": "firecrawl-py not installed"}
        except Exception as e:
            logger.error(f"Error crawling website: {e}", exc_info=True)
            return {"error": f"Error crawling: {str(e)}"}
