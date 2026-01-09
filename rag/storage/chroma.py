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
        """
        Reset cached client.

        Only clears state if client.close() succeeds to prevent resource leaks.
        """
        with self._lock:
            # Explicitly close client if available
            if self._client is not None:
                try:
                    if hasattr(self._client, 'close'):
                        self._client.close()
                    # Only clear state if close succeeds
                    self._client = None
                    self._collection = None
                    self._current_project = None
                except Exception as e:
                    # If close fails, don't clear state - keeps reference for retry
                    logger.error(f"Failed to close ChromaDB client, state preserved: {e}")
                    raise  # Re-raise to signal failure
            else:
                # No client, just clear state
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

        Only resets cache if the deleted project is the current project.
        """
        with self._lock:
            try:
                is_current_project = (self._current_project == project)

                # Get client for the project
                client, _ = self.get_client(project)
                client.delete_collection(project)

                # Only reset if we deleted the current project
                if is_current_project:
                    # Clear state directly since we're in lock
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
