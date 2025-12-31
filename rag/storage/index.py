"""
LlamaIndex storage management.

Handles index creation, loading, and persistence.
"""

import threading
import os
from pathlib import Path
from typing import Optional, List

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings as LlamaSettings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.config import settings, logger
from rag.embeddings import get_embedding_model
from rag.models import CacheStats
from rag.storage.chroma import get_chroma_manager
from rag.utils.timing import log_timing


def _fsync_directory(path: Path) -> None:
    """fsync a directory, ensuring the fd is closed even on error."""
    dir_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _fsync_json_files(path: Path) -> None:
    """fsync json files in a directory; logs and continues on per-file failure."""
    for file_path in path.glob("*.json"):
        try:
            with open(file_path, "r+") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"Failed to fsync {file_path}: {e}")


class IndexManager:
    """
    Manages LlamaIndex instances with project isolation.
    
    Handles index creation, loading, persistence, and caching.
    """
    
    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._current_project: Optional[str] = None
        self._stats = CacheStats()
        self._initialized = False
        self._lock = threading.RLock()
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self, category: Optional[str] = None) -> None:
        """Reset cache statistics."""
        self._stats.reset(category)
    
    @property
    def current_project(self) -> str:
        """Get current active project."""
        return self._current_project or settings.default_project
    
    def _configure_llama_settings(self) -> None:
        """Configure LlamaIndex global settings."""
        if self._initialized:
            return
        
        LlamaSettings.embed_model = get_embedding_model()
        LlamaSettings.text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        self._initialized = True
    
    def get_index(self, project: Optional[str] = None) -> VectorStoreIndex:
        """
        Get or load index for a project.

        Args:
            project: Project name. If None, uses current project.

        Returns:
            VectorStoreIndex instance
        """
        target_project = project or self._current_project or settings.default_project

        with self._lock:
            # Check cache
            if (self._index is not None
                and self._current_project == target_project):
                self._stats.index_cache_hits += 1
                logger.debug(f"Index cache hit for {target_project}")
                return self._index

            self._stats.index_loads += 1
            logger.info(f"Loading index for project: {target_project}")

            # Configure LlamaIndex
            self._configure_llama_settings()

            # Get ChromaDB
            chroma_manager = get_chroma_manager()
            _, collection = chroma_manager.get_client(target_project)
            vector_store = ChromaVectorStore(chroma_collection=collection)

            # Get storage path
            project_path = settings.storage_path / target_project
            project_path.mkdir(parents=True, exist_ok=True)
            docstore_path = project_path / "docstore.json"

            if docstore_path.exists():
                # Load existing index
                with log_timing("index:load", project=target_project):
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store,
                        persist_dir=str(project_path)
                    )
                    self._index = load_index_from_storage(storage_context)
            else:
                # Create new empty index
                with log_timing("index:create", project=target_project):
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self._index = VectorStoreIndex([], storage_context=storage_context)
                    self._index.storage_context.persist(persist_dir=str(project_path))

            self._current_project = target_project
            return self._index

    def insert_nodes(
        self,
        nodes: List,
        project: Optional[str] = None,
        show_progress: bool = False
    ) -> int:
        """
        Insert nodes into the index.
        
        Args:
            nodes: List of nodes to insert
            project: Target project
            show_progress: Show progress bar
            
        Returns:
            Number of nodes inserted
        """
        index = self.get_index(project)
        index.insert_nodes(nodes, show_progress=show_progress)
        return len(nodes)
    
    def persist(self, project: Optional[str] = None) -> None:
        """Persist the index to disk."""
        target_project = project or self._current_project or settings.default_project
        project_path = settings.storage_path / target_project

        with self._lock:
            if self._index is not None:
                with log_timing("index:persist", project=target_project):
                    self._index.storage_context.persist(persist_dir=str(project_path))

                    # Ensure persistence to disk
                    try:
                        _fsync_directory(project_path)
                        _fsync_json_files(project_path)
                    except Exception as e:
                        logger.warning(f"Failed to fsync index files: {e}")

                logger.debug(f"Persisted index for {target_project}")

    def reset(self) -> None:
        """Reset the cached index."""
        with self._lock:
            self._index = None
            self._current_project = None

    def switch_project(self, project: str) -> VectorStoreIndex:
        """
        Switch to a different project.

        Args:
            project: Project name to switch to

        Returns:
            VectorStoreIndex for the new project
        """
        with self._lock:
            self.reset()
            get_chroma_manager().reset()
            return self.get_index(project)

    def get_node_count(self, project: Optional[str] = None) -> int:
        """Get the number of nodes in the index."""
        index = self.get_index(project)
        try:
            docstore = getattr(index, "docstore", None)
            docs = getattr(docstore, "docs", {}) or {}
            count = len(docs)
            if count > 0:
                return count
        except Exception:
            pass

        # Fallback to ChromaDB count
        return get_chroma_manager().get_collection_count(project)

    def get_retriever(
        self,
        project: Optional[str] = None,
        similarity_top_k: int = None
    ):
        """
        Get a retriever for the index.

        Args:
            project: Project name
            similarity_top_k: Number of results to retrieve

        Returns:
            LlamaIndex retriever
        """
        top_k = similarity_top_k or settings.default_top_k
        index = self.get_index(project)
        return index.as_retriever(similarity_top_k=top_k)

    def validate_index(self, project: Optional[str] = None) -> dict:
        """
        Validate the health and integrity of an index.

        Args:
            project: Project name to validate. If None, uses current project.

        Returns:
            Dict with validation results including:
            - healthy: bool indicating overall health
            - node_count: Number of nodes in index
            - chroma_count: Number of documents in ChromaDB
            - storage_exists: Whether storage files exist
            - index_accessible: Whether index can be accessed
            - counts_match: Whether docstore and ChromaDB counts match
            - errors: List of any errors encountered
        """
        target_project = project or self._current_project or settings.default_project
        result = {
            "healthy": True,
            "node_count": 0,
            "chroma_count": 0,
            "storage_exists": False,
            "index_accessible": False,
            "counts_match": False,
            "errors": []
        }

        try:
            # Check storage existence
            project_path = settings.storage_path / target_project
            result["storage_exists"] = project_path.exists() and project_path.is_dir()

            if not result["storage_exists"]:
                result["errors"].append(f"Storage directory not found: {project_path}")
                result["healthy"] = False
                return result

            # Check if index is accessible
            try:
                index = self.get_index(target_project)
                result["index_accessible"] = True
            except Exception as e:
                result["errors"].append(f"Failed to load index: {e}")
                result["healthy"] = False
                return result

            # Get node count from docstore
            try:
                docstore = getattr(index, "docstore", None)
                docs = getattr(docstore, "docs", {}) or {}
                result["node_count"] = len(docs)
            except Exception as e:
                result["errors"].append(f"Failed to get docstore count: {e}")

            # Get count from ChromaDB
            try:
                result["chroma_count"] = get_chroma_manager().get_collection_count(target_project)
            except Exception as e:
                result["errors"].append(f"Failed to get ChromaDB count: {e}")

            # Check if counts match
            if result["node_count"] > 0 and result["chroma_count"] > 0:
                result["counts_match"] = abs(result["node_count"] - result["chroma_count"]) <= 1
                if not result["counts_match"]:
                    result["healthy"] = False
                    result["errors"].append(
                        f"Count mismatch: docstore={result['node_count']}, chroma={result['chroma_count']}"
                    )

            # Final health assessment
            if result["node_count"] == 0 and result["chroma_count"] == 0:
                result["healthy"] = False
                result["errors"].append("Index appears to be empty")

        except Exception as e:
            result["healthy"] = False
            result["errors"].append(f"Validation failed: {e}")
            logger.error(f"Index validation error for {target_project}: {e}")

        return result


# Global singleton instance
_index_manager: Optional[IndexManager] = None
_index_manager_lock = threading.Lock()


def get_index_manager() -> IndexManager:
    """Get the global IndexManager instance."""
    global _index_manager
    if _index_manager is None:
        with _index_manager_lock:
            if _index_manager is None:
                _index_manager = IndexManager()
    return _index_manager
