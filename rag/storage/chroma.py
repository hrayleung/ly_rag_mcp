"""
ChromaDB client management.

Handles ChromaDB client lifecycle with proper isolation per project.
"""

import chromadb
from chromadb import Settings as ChromaSettings
from typing import Optional, Tuple
import threading

from rag.config import settings, logger
from rag.models import CacheStats


class ChromaManager:
    """
    Manages ChromaDB client connections with project isolation.

    Each project gets its own chroma_db folder to ensure data isolation.
    """

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._current_project: Optional[str] = None
        self._stats = CacheStats()
        self._lock = threading.RLock()

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
    ) -> Tuple[chromadb.PersistentClient, chromadb.Collection]:
        """
        Get ChromaDB client and collection for a project.

        Args:
            project: Project name. If None, uses current project.

        Returns:
            Tuple of (client, collection)
        """
        target_project = project or settings.default_project

        with self._lock:
            # Check cache
            if (self._client is not None
                and self._current_project == target_project):
                self._stats.chroma_cache_hits += 1
                return self._client, self._collection

            self._stats.chroma_loads += 1
            logger.info(f"Initializing ChromaDB for project: {target_project}")

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
        """Reset cached client."""
        with self._lock:
            # Explicitly close client if available
            if self._client is not None:
                try:
                    if hasattr(self._client, 'close'):
                        self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing ChromaDB client: {e}")
            self._client = None
            self._collection = None
            self._current_project = None

    def get_collection_count(self, project: Optional[str] = None) -> int:
        """Get document count for a project."""
        _, collection = self.get_client(project)
        return collection.count()

    def delete_collection(self, project: str) -> bool:
        """Delete a project's collection."""
        try:
            client, _ = self.get_client(project)
            client.delete_collection(project)
            # Only reset on success to maintain consistent state
            self.reset()
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
