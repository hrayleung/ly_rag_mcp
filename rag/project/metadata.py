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
from datetime import datetime, timezone
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

# Get current process ID for lock file ownership tracking
try:
    _CURRENT_PID = os.getpid()
except Exception:
    _CURRENT_PID = None

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
STALE_LOCK_THRESHOLD_SECONDS = 300  # 5 minutes


def _acquire_file_lock(lock_handle, path: Path) -> bool:
    """
    Cross-platform file lock with retry logic.

    Returns True if lock acquired, False if locking unavailable.
    """
    max_retries = 3
    retry_delay = 0.1

    for attempt in range(max_retries):
        if fcntl and os.name != "nt":
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as e:
                if e.errno == 11 and attempt < max_retries - 1:  # EAGAIN
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                logger.warning(f"Failed to lock {path}: {e}")
        elif msvcrt and os.name == "nt":
            try:
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, _WINDOWS_LOCK_LENGTH)
                return True
            except OSError as e:
                if e.errno == 33 and attempt < max_retries - 1:  # ERROR_LOCK_VIOLATION
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                logger.warning(f"Failed to lock {path}: {e}")
        else:
            logger.warning(
                f"File locking unavailable on this platform ({os.name}). "
                f"Concurrent access to {path} is not protected."
            )
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
def _file_lock(path: Path, force_break_stale_locks: bool = False):
    """
    Context manager to create a lock file for safe concurrent access.

    Raises TimeoutError if lock cannot be acquired after retries.
    Stale locks (>5 minutes old) are automatically broken.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    locked = False
    lock_handle = None
    try:
        # Check for stale lock and break if necessary
        if lock_path.exists() and force_break_stale_locks:
            try:
                lock_stat = lock_path.stat()
                lock_age = time.time() - lock_stat.st_mtime
                if lock_age > STALE_LOCK_THRESHOLD_SECONDS:
                    logger.info(f"Breaking stale lock for {path} (age: {lock_age:.1f}s)")
                    lock_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Failed to check/break stale lock {lock_path}: {e}")

        lock_handle = lock_path.open("w+b")

        # Write lock metadata for stale detection
        try:
            lock_data = json.dumps({
                "timestamp": time.time(),
                "pid": _CURRENT_PID,
                "path": str(path)
            })
            lock_handle.write(lock_data.encode("utf-8"))
            lock_handle.flush()
            os.fsync(lock_handle.fileno())
        except (OSError, AttributeError) as e:
            logger.warning(f"Failed to write lock metadata: {e}")

        for attempt in range(LOCK_RETRY_ATTEMPTS):
            locked = _acquire_file_lock(lock_handle, lock_path)
            if locked:
                break
            time.sleep(LOCK_RETRY_DELAY * (2 ** attempt))

        if not locked:
            raise TimeoutError(f"Could not acquire lock for {path}")

        yield

    except (OSError, TimeoutError) as e:
        logger.warning(f"Lock operation failed for {path}: {e}")
        raise

    finally:
        if lock_handle:
            if locked:
                _release_file_lock(lock_handle, lock_path, locked)
            try:
                lock_handle.close()
            except OSError:
                pass

        try:
            lock_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to delete lock file {lock_path}: {e}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomically write JSON to disk using temp file and os.replace.

    Ensures durability via flush + fsync. Raises OSError on failure.
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

        # Fsync parent directory for durability
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as e:
            logger.error(f"Failed to fsync parent directory for {path}: {e}")
            raise

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
                            # Bug M7 fix: Only cache if successfully loaded from disk
                            self._cache[project] = metadata
                            return metadata
                except PermissionError as e:
                    # Bug M8 fix: Provide actionable guidance for permission errors
                    logger.error(
                        f"Permission denied reading metadata for {project}: {e}\n"
                        f"File: {meta_path}\n"
                        f"Check file permissions and ensure read access is available."
                    )
                    # Don't cache - let caller use default metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {project}: {e}")
                    # Bug M7 fix: Don't cache corrupted data, use default

            # Cache only valid/default metadata
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
            with _file_lock(meta_path):
                _atomic_write_json(meta_path, metadata.to_dict())
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

    def track_chat_turn(self, project: str) -> None:
        """
        Record a chat turn/activity in metadata.

        Updates last_chat_time to current UTC timestamp and increments
        chat_turn_count. Uses file locking for thread safety.

        Args:
            project: Project name

        Example:
            >>> mm = get_metadata_manager()
            >>> mm.track_chat_turn("my_project")
            >>> metadata = mm.load("my_project")
            >>> assert metadata.chat_turn_count == 1
            >>> assert metadata.last_chat_time is not None
        """
        with self._lock:
            metadata = self.load(project)
            metadata.last_chat_time = datetime.now(timezone.utc).isoformat()
            metadata.chat_turn_count = (metadata.chat_turn_count or 0) + 1
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

        with _file_lock(manifest_path):
            _atomic_write_json(manifest_path, manifest)

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
