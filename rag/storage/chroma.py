"""
ChromaDB client management.

Handles ChromaDB client lifecycle with proper isolation per project.
Supports both local (PersistentClient) and remote (HttpClient) connections.
"""

import chromadb
from chromadb import Settings as ChromaSettings
from typing import Optional, Tuple, Union
import threading

from rag.config import settings, logger
from rag.models import CacheStats

# Type alias for ChromaDB client (can be either local or remote)
ChromaClient = Union[chromadb.PersistentClient, chromadb.HttpClient]


class ChromaManager:
    """
    Manages ChromaDB client connections with project isolation.

    Supports two modes:
    - Local: Each project gets its own chroma_db folder (PersistentClient)
    - Remote: Connects to a ChromaDB server via HTTP (HttpClient)

    Remote mode is enabled by setting CHROMA_HOST environment variable.
    """

    def __init__(self):
        self._client: Optional[ChromaClient] = None
        self._collection = None
        self._current_project: Optional[str] = None
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._is_remote = settings.use_remote_chroma

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self, category: Optional[str] = None) -> None:
        """Reset cache statistics."""
        self._stats.reset(category)

    def get_client(
        self,
        project: Optional[str] = None
    ) -> Tuple[ChromaClient, chromadb.Collection]:
        """
        Get ChromaDB client and collection for a project.

        Args:
            project: Project name. If None, uses current project.

        Returns:
            Tuple of (client, collection)
        """
        target_project = project or settings.default_project

        with self._lock:
            # For remote mode, we reuse the same client but switch collections
            if self._is_remote:
                return self._get_remote_client(target_project)
            else:
                return self._get_local_client(target_project)

    def _get_remote_client(
        self,
        target_project: str
    ) -> Tuple[chromadb.HttpClient, chromadb.Collection]:
        """Get or create remote HttpClient connection."""
        # For remote, we keep one client but can switch collections
        if self._client is not None:
            # Check if we need a different collection
            if self._current_project == target_project:
                self._stats.chroma_cache_hits += 1
                return self._client, self._collection

            # Same client, different collection
            self._collection = self._client.get_or_create_collection(target_project)
            self._current_project = target_project
            logger.info(f"Switched to collection: {target_project}")
            return self._client, self._collection

        # Create new remote client
        self._stats.chroma_loads += 1
        logger.info(f"Connecting to remote ChromaDB at {settings.chroma_host}:{settings.chroma_port or 8000}")

        chroma_settings = ChromaSettings(
            anonymized_telemetry=False
        )

        # Build connection kwargs
        connect_kwargs = {
            "host": settings.chroma_host,
            "port": settings.chroma_port or 8000,
            "ssl": settings.chroma_ssl,
            "settings": chroma_settings,
            "tenant": settings.chroma_tenant,
            "database": settings.chroma_database,
        }

        # Add auth header if API key is provided
        if settings.chroma_api_key:
            connect_kwargs["headers"] = {
                "Authorization": f"Bearer {settings.chroma_api_key}"
            }

        self._client = chromadb.HttpClient(**connect_kwargs)
        self._collection = self._client.get_or_create_collection(target_project)
        self._current_project = target_project

        logger.info(f"Connected to remote ChromaDB, collection: {target_project}")
        return self._client, self._collection

    def _get_local_client(
        self,
        target_project: str
    ) -> Tuple[chromadb.PersistentClient, chromadb.Collection]:
        """Get or create local PersistentClient."""
        # Check if project switched - if so, close old client to prevent leaks
        if self._client is not None and self._current_project != target_project:
            logger.info(f"Project switched from {self._current_project} to {target_project}, closing old ChromaDB client")
            try:
                if hasattr(self._client, 'close'):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client during switch: {e}")

            self._client = None
            self._collection = None

        # Check cache
        if (self._client is not None
            and self._current_project == target_project):
            self._stats.chroma_cache_hits += 1
            return self._client, self._collection

        self._stats.chroma_loads += 1
        logger.info(f"Initializing local ChromaDB for project: {target_project}")

        # Build path: ./storage/{project}/chroma_db
        chroma_path = settings.storage_path / target_project / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)

        # Configure and create client
        chroma_settings = ChromaSettings(
            allow_reset=True,
            anonymized_telemetry=False
        )

        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=chroma_settings
        )

        self._collection = self._client.get_or_create_collection(target_project)
        self._current_project = target_project

        return self._client, self._collection

    def reset(self) -> None:
        """
        Reset cached client.

        For local mode: closes client if possible to prevent resource leaks.
        For remote mode: just clears references (connection is stateless HTTP).
        """
        with self._lock:
            if self._client is not None:
                # Only close for local PersistentClient
                if not self._is_remote:
                    try:
                        if hasattr(self._client, 'close'):
                            self._client.close()
                    except Exception as e:
                        logger.error(f"Failed to close ChromaDB client, state preserved: {e}")
                        raise

            # Clear state
            self._client = None
            self._collection = None
            self._current_project = None

    def get_collection_count(self, project: Optional[str] = None) -> int:
        """Get document count for a project."""
        _, collection = self.get_client(project)
        return collection.count()

    def delete_collection(self, project: str) -> bool:
        """
        Delete a project's collection.

        For local mode: resets cache if the deleted project is current.
        For remote mode: just clears collection reference if current.
        """
        with self._lock:
            try:
                is_current_project = (self._current_project == project)

                # Get client for the project
                client, _ = self.get_client(project)
                client.delete_collection(project)

                # Only reset if we deleted the current project
                if is_current_project:
                    if self._is_remote:
                        # Remote: keep client, clear collection
                        self._collection = None
                        self._current_project = None
                        logger.info(f"Deleted collection: {project} (remote)")
                    else:
                        # Local: close client and clear all state
                        if self._client is not None:
                            try:
                                if hasattr(self._client, 'close'):
                                    self._client.close()
                            except Exception as e:
                                logger.warning(f"Error closing ChromaDB client: {e}")
                        self._client = None
                        self._collection = None
                        self._current_project = None
                        logger.info(f"Deleted current project collection: {project}, cache reset")
                else:
                    logger.info(f"Deleted other project collection: {project}, cache preserved")

                return True
            except Exception as e:
                logger.error(f"Failed to delete collection {project}: {e}")
                return False


# Global singleton instance with thread-safe initialization
_chroma_manager: Optional[ChromaManager] = None
_chroma_manager_lock = threading.Lock()


def get_chroma_manager() -> ChromaManager:
    """Get the global ChromaManager instance."""
    global _chroma_manager
    if _chroma_manager is None:
        with _chroma_manager_lock:
            if _chroma_manager is None:
                _chroma_manager = ChromaManager()
    return _chroma_manager
