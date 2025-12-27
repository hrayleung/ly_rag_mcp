"""
Unit tests for rag/project/metadata.py
"""

import json
import os
import tempfile
import threading
import time
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.config import settings
from rag.models import ProjectMetadata
from rag.project.metadata import (
    MetadataManager,
    _atomic_write_json,
    _file_lock,
    _acquire_file_lock,
    _release_file_lock,
    get_metadata_manager,
)


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage directory."""
    original_storage = settings.storage_path
    settings.storage_path = tmp_path
    yield tmp_path
    settings.storage_path = original_storage


@pytest.fixture
def manager(temp_storage):
    """Create a MetadataManager instance with temporary storage."""
    mgr = MetadataManager()
    yield mgr
    mgr.clear_cache()


class TestAtomicWriteJson:
    """Tests for _atomic_write_json function."""

    def test_basic_write(self, temp_storage):
        """Test basic atomic write operation."""
        test_path = temp_storage / "test.json"
        payload = {"key": "value", "number": 42}

        _atomic_write_json(test_path, payload)

        assert test_path.exists()
        with test_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == payload

    def test_temp_file_cleanup_on_success(self, temp_storage):
        """Test that temp files are cleaned up after successful write."""
        test_path = temp_storage / "cleanup_test.json"
        payload = {"test": "data"}

        _atomic_write_json(test_path, payload)

        # Check no .tmp files remain
        tmp_files = list(temp_storage.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_temp_file_cleanup_on_write_failure(self, temp_storage):
        """Test that temp files are cleaned up even when write fails."""
        test_path = temp_storage / "failure_test.json"

        # Mock json.dump to raise an error
        with patch("rag.project.metadata.json.dump", side_effect=OSError("Write failed")):
            with pytest.raises(OSError):
                _atomic_write_json(test_path, {"test": "data"})

        # Temp file should be cleaned up
        tmp_files = list(temp_storage.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_parent_directory_creation(self, temp_storage):
        """Test that parent directories are created if missing."""
        nested_path = temp_storage / "nested" / "deep" / "file.json"
        assert not nested_path.parent.exists()

        _atomic_write_json(nested_path, {"created": True})

        assert nested_path.exists()

    def test_write_with_special_characters(self, temp_storage):
        """Test writing JSON with special characters."""
        test_path = temp_storage / "special.json"
        payload = {
            "text": "Hello\nWorld\t\"Quotes\"",
            "unicode": "Emojis: 🎉 中文",
        }

        _atomic_write_json(test_path, payload)

        with test_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["text"] == payload["text"]
        assert data["unicode"] == payload["unicode"]


class TestFileLock:
    """Tests for file locking functionality."""

    def test_lock_acquisition_success(self, temp_storage):
        """Test successful file lock acquisition."""
        lock_path = temp_storage / "lock_test.txt"
        lock_path.touch()

        with open(lock_path, "a+b") as handle:
            result = _acquire_file_lock(handle, lock_path)
            # Should succeed on Unix with fcntl
            if os.name != "nt":
                assert result is True
            _release_file_lock(handle, lock_path, result)

    def test_lock_release(self, temp_storage):
        """Test that lock is properly released."""
        lock_path = temp_storage / "release_test.txt"
        lock_path.touch()

        with open(lock_path, "a+b") as handle:
            locked = _acquire_file_lock(handle, lock_path)
            _release_file_lock(handle, lock_path, locked)

        # Should be able to acquire again immediately
        with open(lock_path, "a+b") as handle:
            result = _acquire_file_lock(handle, lock_path)
            if os.name != "nt":
                assert result is True
            _release_file_lock(handle, lock_path, result)

    def test_file_lock_context_manager(self, temp_storage):
        """Test _file_lock context manager."""
        test_file = temp_storage / "context_test.json"
        test_file.write_text("{}")

        with _file_lock(test_file):
            # File should be locked (we're inside the context)
            pass
        # Context exited, lock released


class TestMetadataManagerLoad:
    """Tests for MetadataManager.load method."""

    def test_load_nonexistent_project(self, manager, temp_storage):
        """Test loading metadata for a project that doesn't exist."""
        result = manager.load("nonexistent")

        assert isinstance(result, ProjectMetadata)
        assert result.name == "nonexistent"

    def test_load_existing_metadata(self, manager, temp_storage):
        """Test loading existing metadata from disk."""
        project = "existing_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = ProjectMetadata(
            name=project,
            display_name="Existing Project",
            description="A test project"
        )
        meta_path.write_text(json.dumps(metadata.to_dict()))

        # Load it
        result = manager.load(project)

        assert result.name == project
        assert result.display_name == "Existing Project"
        assert result.description == "A test project"

    def test_load_caches_result(self, manager, temp_storage):
        """Test that load caches the result."""
        project = "cached_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"name": project, "display_name": "Test"}))

        # First load
        result1 = manager.load(project)
        # Modify file
        meta_path.write_text(json.dumps({"name": project, "display_name": "Changed"}))

        # Second load should return cached
        result2 = manager.load(project)

        assert result1.display_name == result2.display_name

    def test_backward_compatibility_load_missing_fields(self, manager, temp_storage):
        """Test loading metadata with missing optional fields (backward compat)."""
        project = "legacy_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Create legacy metadata with only required fields
        legacy_data = {"name": project}
        meta_path.write_text(json.dumps(legacy_data))

        result = manager.load(project)

        assert result.name == project
        assert result.display_name == project  # Defaults to name
        assert result.keywords == []
        assert result.default_paths == []

    def test_backward_compatibility_load_non_dict_json(self, manager, temp_storage):
        """Test loading metadata when JSON is not a dict (legacy format)."""
        project = "weird_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Write non-dict JSON
        meta_path.write_text('["legacy", "format"]')

        result = manager.load(project)

        # Should fall back to defaults
        assert result.name == project

    def test_backward_compatibility_load_empty_file(self, manager, temp_storage):
        """Test loading metadata from empty file."""
        project = "empty_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("")

        result = manager.load(project)

        assert result.name == project

    def test_path_traversal_prevention(self, manager, temp_storage):
        """Test that path traversal attempts are blocked."""
        # Try to access outside storage path
        result = manager.load("../../../etc/passwd")

        # Should return default metadata, not the external file
        assert result.name == "../../../etc/passwd"

    def test_corrupted_json_load(self, manager, temp_storage):
        """Test loading from corrupted JSON file."""
        project = "corrupted_project"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("{ invalid json }")

        result = manager.load(project)

        assert result.name == project  # Falls back to default


class TestMetadataManagerSave:
    """Tests for MetadataManager.save method."""

    def test_save_creates_file(self, manager, temp_storage):
        """Test that save creates the metadata file."""
        project = "new_project"
        metadata = ProjectMetadata(name=project, description="Test description")

        result = manager.save(project, metadata)

        meta_path = settings.storage_path / project / settings.project_metadata_filename
        assert meta_path.exists()
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["name"] == project
        assert data["description"] == "Test description"

    def test_save_updates_timestamps(self, manager, temp_storage):
        """Test that save updates timestamps."""
        project = "timestamp_project"
        metadata = ProjectMetadata(name=project)
        original_created = metadata.created_at

        time.sleep(0.01)  # Ensure different timestamp
        result = manager.save(project, metadata)

        assert result.updated_at is not None
        assert result.created_at == original_created  # Should not change

    def test_save_with_initialize_flag(self, manager, temp_storage):
        """Test save with initialize=True sets empty description."""
        project = "initialize_project"

        result = manager.save(project, initialize=True)

        assert result.description == ""

    def test_save_lock_failure_warning(self, manager, temp_storage, caplog):
        """Test that save handles lock failures gracefully."""
        project = "lock_fail_project"

        # Mock _file_lock to fail
        with patch("rag.project.metadata._file_lock", side_effect=OSError("Lock failed")):
            metadata = ProjectMetadata(name=project, description="Test")
            result = manager.save(project, metadata)

        # Should still return metadata (with warning logged)
        assert result.name == project


class TestConcurrentSaves:
    """Tests for concurrent save operations."""

    def test_concurrent_saves_from_multiple_threads(self, temp_storage):
        """Test that concurrent saves from multiple threads work correctly."""
        project = "concurrent_project"
        errors = []

        def save_metadata(thread_id):
            try:
                manager = MetadataManager()
                metadata = ProjectMetadata(
                    name=project,
                    description=f"Description from thread {thread_id}"
                )
                manager.save(project, metadata)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=save_metadata, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify file exists and has valid content
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        assert meta_path.exists()
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["name"] == project


class TestManifestMethods:
    """Tests for manifest load/save methods."""

    def test_load_manifest_nonexistent(self, manager, temp_storage):
        """Test loading manifest that doesn't exist."""
        result = manager.load_manifest("nonexistent_manifest")

        assert isinstance(result, dict)
        assert "roots" in result
        assert "updated_at" in result

    def test_load_manifest_existing(self, manager, temp_storage):
        """Test loading existing manifest."""
        project = "test_project"
        manifest_path = settings.storage_path / project / settings.ingest_manifest_filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        existing_manifest = {
            "roots": {"file1.py": {"hash": "abc123"}},
            "updated_at": "2024-01-01T00:00:00"
        }
        manifest_path.write_text(json.dumps(existing_manifest))

        result = manager.load_manifest(project)

        assert result["roots"] == {"file1.py": {"hash": "abc123"}}

    def test_save_manifest(self, manager, temp_storage):
        """Test saving manifest."""
        project = "manifest_project"
        manifest = {"roots": {"test.txt": {"hash": "xyz"}}}

        manager.save_manifest(project, manifest)

        manifest_path = settings.storage_path / project / settings.ingest_manifest_filename
        assert manifest_path.exists()
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["roots"] == {"test.txt": {"hash": "xyz"}}
        assert "updated_at" in data

    def test_manifest_path_traversal_prevention(self, manager, temp_storage):
        """Test that manifest loading blocks path traversal."""
        result = manager.load_manifest("../../../etc/passwd")

        assert result["roots"] == {}  # Returns default

    def test_save_manifest_lock_failure(self, manager, temp_storage, caplog):
        """Test that save_manifest handles lock failures gracefully."""
        project = "manifest_lock_fail"

        with patch("rag.project.metadata._file_lock", side_effect=OSError("Lock failed")):
            # Should not raise, just log warning
            manager.save_manifest(project, {"roots": {}})


class TestClearCache:
    """Tests for cache clearing functionality."""

    def test_clear_specific_project_cache(self, manager, temp_storage):
        """Test clearing cache for specific project."""
        project = "cache_test"
        meta_path = settings.storage_path / project / settings.project_metadata_filename
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"name": project}))

        manager.load(project)
        manager.clear_cache(project)

        assert project not in manager._cache

    def test_clear_all_cache(self, manager, temp_storage):
        """Test clearing all cache."""
        for i in range(3):
            project = f"cache_test_{i}"
            meta_path = settings.storage_path / project / settings.project_metadata_filename
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({"name": project}))
            manager.load(project)

        manager.clear_cache()

        assert len(manager._cache) == 0


class TestGlobalSingleton:
    """Tests for the global MetadataManager singleton."""

    def test_get_metadata_manager_returns_singleton(self):
        """Test that get_metadata_manager returns the same instance."""
        # Reset singleton
        import rag.project.metadata
        rag.project.metadata._metadata_manager = None

        manager1 = get_metadata_manager()
        manager2 = get_metadata_manager()

        assert manager1 is manager2
