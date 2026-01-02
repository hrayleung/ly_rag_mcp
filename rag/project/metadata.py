"""
Project metadata management.
"""

import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import settings, logger
from rag.models import ProjectMetadata

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

# Common stop words to exclude from keywords
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'src', 'lib', 'dist', 'build', 'node_modules', 'test', 'tests',
    'main', 'index', 'app', 'utils', 'helpers', 'common', 'shared',
    '__pycache__', 'venv', 'env', '.git', '.idea', '.vscode'
}

_WINDOWS_LOCK_LENGTH = 0x7fff0000
LOCK_RETRY_ATTEMPTS = settings.lock_retry_attempts
LOCK_RETRY_DELAY = settings.lock_retry_delay


def _acquire_file_lock(lock_handle, path: Path) -> bool:
    """Best-effort cross-platform file lock with retry logic; logs on failure but continues."""
    max_retries = 3
    retry_delay = 0.1  # 100ms

    for attempt in range(max_retries):
        if fcntl and os.name != "nt":
            try:
                # Try non-blocking lock first
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as e:
                if e.errno == 11:  # EAGAIN - would block
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                logger.warning(f"Failed to lock {path}: {e}")
        elif msvcrt and os.name == "nt":
            try:
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, _WINDOWS_LOCK_LENGTH)
                return True
            except OSError as e:
                if e.errno == 33:  # ERROR_LOCK_VIOLATION
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                logger.warning(f"Failed to lock {path}: {e}")
        else:
            # No locking support
            return False

    return False


def _release_file_lock(lock_handle, path: Path, locked: bool) -> None:
    """Release file lock if acquired."""
    if not locked:
        return
    try:
        if fcntl and os.name != "nt":
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        elif msvcrt and os.name == "nt":
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, _WINDOWS_LOCK_LENGTH)
    except OSError as e:
        logger.warning(f"Failed to release lock for {path}: {e}")


@contextmanager
def _file_lock(path: Path):
    """
    Context manager to create a separate lock file and apply a
    best-effort lock; raises TimeoutError if lock fails.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    locked = False
    lock_handle = None
    try:
        # Use a separate lock file
        # w+b is fine because we don't care about content, just the file handle
        lock_handle = lock_path.open("w+b")

        for attempt in range(LOCK_RETRY_ATTEMPTS):
            locked = _acquire_file_lock(lock_handle, lock_path)
            if locked:
                break
            time.sleep(LOCK_RETRY_DELAY * (2 ** attempt))

        if not locked:
            raise TimeoutError(f"Could not acquire lock for {path} (via {lock_path})")

        yield

    except (OSError, TimeoutError) as e:
        logger.warning(f"Lock acquisition failed for {path}: {e}")
        if lock_handle:
            try:
                lock_handle.close()
            except OSError:
                pass
        raise  # Re-raise to prevent unsafe operations

    finally:
        if lock_handle:
            if locked:
                _release_file_lock(lock_handle, lock_path, locked)
            try:
                lock_handle.close()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomically write JSON to disk using a temp file and os.replace,
    ensuring durability via flush + fsync and cleaning temp files on failure.
    """
    temp_path: Optional[Path] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=path.parent, encoding="utf-8", suffix=".tmp"
        ) as tmp:
            temp_path = Path(tmp.name)
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_path, path)
        # Verify the file was written
        if not path.exists():
            raise OSError(f"Failed to write {path}: os.replace() did not create file")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")


class MetadataManager:
    """
    Manages project metadata storage and retrieval.
    """

    def __init__(self):
        self._cache: Dict[str, ProjectMetadata] = {}
        self._lock = threading.RLock()

    def _get_metadata_path(self, project: str) -> Path:
        """Get path to metadata file for a project."""
        return settings.storage_path / project / settings.project_metadata_filename

    def _get_manifest_path(self, project: str) -> Path:
        """Get path to ingest manifest for a project."""
        return settings.storage_path / project / settings.ingest_manifest_filename

    def load(self, project: str) -> ProjectMetadata:
        """
        Load metadata for a project.

        Args:
            project: Project name

        Returns:
            ProjectMetadata instance
        """
        with self._lock:
            if project in self._cache:
                return self._cache[project]

            meta_path = self._get_metadata_path(project)
            metadata = ProjectMetadata(name=project)

            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            metadata = ProjectMetadata.from_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {project}: {e}")

            self._cache[project] = metadata
            return metadata

    def save(
        self,
        project: str,
        metadata: Optional[ProjectMetadata] = None,
        initialize: bool = False
    ) -> ProjectMetadata:
        """
        Save metadata for a project.

        Args:
            project: Project name
            metadata: Metadata to save (creates default if None)
            initialize: If True and no description, set empty string

        Returns:
            Saved metadata
        """
        meta_path = self._get_metadata_path(project)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        if metadata is None:
            metadata = ProjectMetadata(name=project)

        # Update timestamps
        metadata.updated_at = datetime.now().isoformat()
        if not metadata.created_at:
            metadata.created_at = metadata.updated_at

        if initialize and not metadata.description:
            metadata.description = ""

        with self._lock:
            write_success = False
            try:
                with _file_lock(meta_path):
                    _atomic_write_json(meta_path, metadata.to_dict())
                write_success = True
            except Exception as e:
                logger.warning(f"Skipping lock for {meta_path}: {e}")
                try:
                    _atomic_write_json(meta_path, metadata.to_dict())
                    write_success = True
                except Exception as e2:
                    logger.error(f"Failed to write metadata for {project}: {e2}")
                    raise

            # Only update cache after successful write
            if write_success:
                self._cache[project] = metadata
        return metadata

    def ensure_exists(self, project: str) -> ProjectMetadata:
        """Ensure metadata file exists for project."""
        with self._lock:
            meta_path = self._get_metadata_path(project)
            if not meta_path.exists():
                return self.save(project, initialize=True)
            return self.load(project)

    def update(
        self,
        project: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        default_paths: Optional[List[str]] = None
    ) -> ProjectMetadata:
        """
        Update specific metadata fields.

        Args:
            project: Project name
            display_name: New display name
            description: New description
            keywords: New keywords list
            default_paths: New default paths

        Returns:
            Updated metadata
        """
        metadata = self.load(project)

        if display_name is not None:
            metadata.display_name = display_name.strip() or project

        if description is not None:
            metadata.description = description.strip()

        if keywords is not None:
            metadata.keywords = self._normalize_list(keywords)

        if default_paths is not None:
            metadata.default_paths = self._normalize_list(default_paths)

        return self.save(project, metadata)

    def record_index_activity(
        self,
        project: str,
        source_hint: Optional[str] = None
    ) -> None:
        """Record indexing activity in metadata."""
        with self._lock:
            metadata = self.load(project)
            metadata.last_indexed = datetime.now().isoformat()

            if source_hint:
                paths = metadata.default_paths or []
                if source_hint not in paths:
                    paths = (paths + [source_hint])[-5:]  # Keep last 5
                    metadata.default_paths = paths

            self.save(project, metadata)

    def _normalize_list(self, value: Any) -> List[str]:
        """Normalize various inputs to list of strings."""
        if value is None:
            return []

        if isinstance(value, str):
            items = re.split(r"[,\n]", value)
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return [str(item).strip() for item in items if str(item).strip()]

    # Manifest methods for incremental indexing

    def load_manifest(self, project: str) -> dict:
        """Load ingest manifest for incremental tracking."""
        manifest_path = self._get_manifest_path(project)

        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data.setdefault("roots", {})
                        return data
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

        return {"roots": {}, "updated_at": datetime.now().isoformat()}

    def save_manifest(self, project: str, manifest: dict) -> None:
        """Save ingest manifest."""
        manifest_path = self._get_manifest_path(project)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest["updated_at"] = datetime.now().isoformat()

        write_success = False
        try:
            with _file_lock(manifest_path):
                _atomic_write_json(manifest_path, manifest)
            write_success = True
        except Exception as e:
            logger.warning(f"Skipping lock for {manifest_path}: {e}")
            try:
                _atomic_write_json(manifest_path, manifest)
                write_success = True
            except Exception as e2:
                logger.error(f"Failed to write manifest: {e2}")
                raise

    def clear_cache(self, project: Optional[str] = None) -> None:
        """Clear metadata cache."""
        if project:
            self._cache.pop(project, None)
        else:
            self._cache.clear()

    def extract_keywords_from_directory(self, dir_path: Path, max_keywords: int = 15) -> List[str]:
        """
        Extract keywords from directory name and file names.

        Args:
            dir_path: Directory path to analyze
            max_keywords: Maximum number of keywords to return
        """
        words = Counter()

        # Directory name has higher weight
        dir_name = dir_path.name.lower()
        for word in re.split(r'[-_\s]+', dir_name):
            if len(word) > 2 and word not in STOP_WORDS:
                words[word] += 5

        # Scan files
        try:
            for file in dir_path.rglob("*"):
                if not file.is_file():
                    continue
                # Skip hidden files and common junk directories
                parts = file.relative_to(dir_path).parts
                if any(p.startswith('.') or p in STOP_WORDS for p in parts):
                    continue

                # File name (without extension)
                stem = file.stem.lower()
                for word in re.split(r'[-_\s]+', stem):
                    if len(word) > 2 and word not in STOP_WORDS:
                        words[word] += 1

                # Extension as tech stack indicator
                ext = file.suffix.lower()
                ext_map = {
                    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                    '.java': 'java', '.go': 'golang', '.rs': 'rust',
                    '.cpp': 'cpp', '.c': 'c', '.rb': 'ruby',
                    '.md': 'markdown', '.pdf': 'pdf', '.docx': 'word'
                }
                if ext in ext_map:
                    words[ext_map[ext]] += 2
        except Exception as e:
            logger.warning(f"Error scanning directory: {e}")

        return [word for word, _ in words.most_common(max_keywords)]

    def learn_from_query(self, project: str, query: str, success: bool = True) -> None:
        """
        Learn from successful queries and update metadata keywords.

        Args:
            project: Project name
            query: Query text
            success: Whether relevant results were found
        """
        if not success:
            return

        metadata = self.load(project)
        existing = set(metadata.keywords or [])

        # Extract keywords from query
        query_words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        new_keywords = [w for w in query_words if w not in STOP_WORDS and w not in existing]

        if new_keywords:
            # Keep at most 30 keywords
            combined = list(existing) + new_keywords[:5]
            metadata.keywords = combined[-30:]
            self.save(project, metadata)
            logger.info(f"Learned keywords for {project}: {new_keywords[:5]}")


# Global singleton with thread-safe initialization
_metadata_manager: Optional[MetadataManager] = None
_metadata_manager_lock = threading.Lock()


def get_metadata_manager() -> MetadataManager:
    """Get the global MetadataManager instance."""
    global _metadata_manager
    if _metadata_manager is None:
        with _metadata_manager_lock:
            if _metadata_manager is None:
                _metadata_manager = MetadataManager()
    return _metadata_manager
