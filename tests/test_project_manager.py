"""Tests for ProjectManager concurrency fixes.

Tests for Bug C3: switch_project() race condition
Tests for Bug C5: create_project() TOCTOU race
Tests for Bug H1: choose_project() thread safety
Tests for Bug H2: discover_projects() thread safety
Tests for Bug H4: infer_project() logging
Tests for Bug H5: current_project property thread safety
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rag.config import settings
from rag.project.manager import get_project_manager, _project_manager, ProjectManager
from rag.project.metadata import get_metadata_manager
from rag.storage.index import _index_manager
from rag.storage.chroma import _chroma_manager


@pytest.fixture()
def temp_storage_settings(tmp_path, monkeypatch):
    """Point rag.settings.storage_path at a temp directory."""
    # Reset singleton before changing storage path
    global _project_manager
    _project_manager = None

    original_storage = settings.storage_path
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    yield tmp_path
    monkeypatch.setattr(settings, "storage_path", original_storage)

    # Reset after test - clear all singleton state
    _project_manager = None

    # Clear any other manager singletons
    from rag.storage.index import _index_manager
    from rag.storage.chroma import _chroma_manager
    _index_manager = None
    _chroma_manager = None


def _write_metadata(storage_root, project: str, payload: dict):
    """Helper to write metadata file."""
    proj_dir = storage_root / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    meta_path = proj_dir / settings.project_metadata_filename
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def _create_project_directories(storage_root, project_names: list[str]):
    """Helper to create project directories with metadata."""
    for project in project_names:
        proj_dir = storage_root / project
        proj_dir.mkdir(parents=True, exist_ok=True)

        # Create chroma_db subdirectory
        chroma_dir = proj_dir / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "name": project,
            "display_name": project,
            "description": f"Test project {project}",
            "keywords": [],
            "default_paths": [],
            "last_indexed": None,
            "created_at": "2026-01-05T00:00:00Z",
            "updated_at": "2026-01-05T00:00:00Z",
        }
        _write_metadata(storage_root, project, metadata)


def test_switch_project_concurrent(temp_storage_settings):
    """Test Bug C3: Concurrent switches don't corrupt state.

    This test verifies that concurrent calls to switch_project()
    maintain consistent state. Before the fix, _current_project
    could be corrupted by race conditions.
    """
    storage = temp_storage_settings

    # Create two projects
    _create_project_directories(storage, ["project_a", "project_b"])

    pm = get_project_manager()
    results = []
    errors = []

    def switch_and_report(proj):
        """Switch to project and record result."""
        try:
            result = pm.switch_project(proj)
            # Record what project we switched to and what state we see
            results.append((proj, pm.current_project, result))
        except Exception as e:
            errors.append((proj, str(e)))

    # Start concurrent switches
    threads = [
        threading.Thread(target=switch_and_report, args=("project_a",)),
        threading.Thread(target=switch_and_report, args=("project_b",)),
        threading.Thread(target=switch_and_report, args=("project_a",)),
        threading.Thread(target=switch_and_report, args=("project_b",)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    # Check no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Check all switches succeeded
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    # Verify final state is consistent
    # After all threads complete, current_project should be one of our projects
    assert pm.current_project in ["project_a", "project_b"], \
        f"Invalid final state: {pm.current_project}"

    # Verify all successful switches maintained consistency
    for target, observed, result in results:
        assert "success" in result, f"Switch to {target} failed: {result}"
        # At the time of observation, current_project should have matched
        # (This is a weak check since state can change between calls)
        assert observed in ["project_a", "project_b"], \
            f"Invalid observed state: {observed}"


def test_switch_project_stress(temp_storage_settings):
    """Stress test with many concurrent switches."""
    storage = temp_storage_settings

    # Create multiple projects
    projects = [f"project_{i}" for i in range(5)]
    _create_project_directories(storage, projects)

    pm = get_project_manager()
    results = []
    errors = []

    def switch_multiple(proj, count):
        """Switch to project multiple times."""
        for _ in range(count):
            try:
                pm.switch_project(proj)
                results.append((proj, pm.current_project))
            except Exception as e:
                errors.append((proj, str(e)))

    # Start many threads switching between projects
    threads = []
    for proj in projects:
        t = threading.Thread(target=switch_multiple, args=(proj, 10))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    # Check no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Check all operations completed
    assert len(results) == 50, f"Expected 50 results, got {len(results)}"

    # Verify final state is valid
    assert pm.current_project in projects


def test_create_project_concurrent(temp_storage_settings):
    """Test Bug C5: Concurrent creation of same project doesn't create duplicates.

    This test verifies the TOCTOU (Time-Of-Check-Time-Of-Use) fix in create_project().
    Before the fix, two threads could pass the exists() check and both create
    the project, leading to inconsistent state.
    """
    storage = temp_storage_settings

    pm = get_project_manager()
    project_name = "test_concurrent_creation"
    results = []
    errors = []

    def create_and_report():
        """Try to create project and record result."""
        try:
            result = pm.create_project(project_name)
            results.append(result)
        except Exception as e:
            errors.append(str(e))

    # Start concurrent creations
    threads = [
        threading.Thread(target=create_and_report),
        threading.Thread(target=create_and_report),
        threading.Thread(target=create_and_report),
        threading.Thread(target=create_and_report),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    # Check no exceptions
    assert len(errors) == 0, f"Exceptions occurred: {errors}"

    # Check all attempts completed
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    # Only one should succeed
    successes = [r for r in results if "success" in r]
    failures = [r for r in results if "error" in r]

    assert len(successes) == 1, f"Expected 1 success, got {len(successes)}: {successes}"
    assert len(failures) == 3, f"Expected 3 failures, got {len(failures)}: {failures}"

    # Verify project was actually created
    assert pm.project_exists(project_name), "Project should exist"

    # Verify directory exists
    project_path = storage / project_name
    assert project_path.exists(), "Project directory should exist"

    # Verify metadata file exists
    metadata_path = project_path / settings.project_metadata_filename
    assert metadata_path.exists(), "Metadata file should exist"


def test_create_project_concurrent_different_names(temp_storage_settings):
    """Test concurrent creation of different projects works correctly."""
    storage = temp_storage_settings

    pm = get_project_manager()
    project_names = ["project_x", "project_y", "project_z"]
    results = []
    errors = []

    def create_and_report(name):
        """Create project and record result."""
        try:
            result = pm.create_project(name)
            results.append((name, result))
        except Exception as e:
            errors.append((name, str(e)))

    # Create projects concurrently
    threads = [
        threading.Thread(target=create_and_report, args=(project_names[0],)),
        threading.Thread(target=create_and_report, args=(project_names[1],)),
        threading.Thread(target=create_and_report, args=(project_names[2],)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    # Check no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # All should succeed
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    for name, result in results:
        assert "success" in result, f"Failed to create {name}: {result}"
        assert result["project"] == name, f"Wrong project name in result"

    # Verify all projects exist
    for name in project_names:
        assert pm.project_exists(name), f"Project {name} should exist"
        project_path = storage / name
        assert project_path.exists(), f"Directory for {name} should exist"


def test_create_project_idempotent(temp_storage_settings):
    """Test that creating same project twice is idempotent."""
    storage = temp_storage_settings

    pm = get_project_manager()
    project_name = "test_idempotent"

    # First creation should succeed
    result1 = pm.create_project(project_name)
    assert "success" in result1, f"First creation failed: {result1}"

    # Second creation should fail gracefully
    result2 = pm.create_project(project_name)
    assert "error" in result2, f"Second creation should fail: {result2}"
    assert "already exists" in result2["error"], f"Wrong error message: {result2['error']}"

    # Verify only one directory exists
    project_path = storage / project_name
    assert project_path.exists(), "Project should exist"

    # Verify metadata is intact
    metadata_path = project_path / settings.project_metadata_filename
    assert metadata_path.exists(), "Metadata should exist"


def test_switch_and_create_concurrent(temp_storage_settings):
    """Test concurrent switch and create operations."""
    storage = temp_storage_settings

    # Create one project to start with
    _create_project_directories(storage, ["existing_project"])

    pm = get_project_manager()
    results = {
        "switch": [],
        "create": [],
        "errors": []
    }

    def switch_projects():
        """Switch between projects."""
        for i in range(10):
            try:
                pm.switch_project("existing_project")
                results["switch"].append(i)
            except Exception as e:
                results["errors"].append(("switch", str(e)))

    def create_projects():
        """Try to create new project."""
        for i in range(10):
            try:
                result = pm.create_project("new_project")
                if "success" in result:
                    results["create"].append(i)
            except Exception as e:
                results["errors"].append(("create", str(e)))

    # Run concurrent operations
    t1 = threading.Thread(target=switch_projects)
    t2 = threading.Thread(target=create_projects)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Check no critical errors
    critical_errors = [e for e in results["errors"] if "FileExistsError" not in str(e)]
    assert len(critical_errors) == 0, f"Critical errors occurred: {critical_errors}"

    # Verify project was created
    assert pm.project_exists("new_project"), "New project should exist"

    # Verify state is consistent
    assert pm.current_project in ["existing_project", "new_project"]


def test_lock_effectiveness(temp_storage_settings):
    """Test that locks actually prevent race conditions.

    This is a meta-test to verify the locking mechanism works.
    """
    storage = temp_storage_settings

    _create_project_directories(storage, ["project_lock_test"])

    pm = get_project_manager()

    # Start from a known state - switch to our test project first
    pm.switch_project("project_lock_test")

    # Track access patterns
    access_log = []
    lock_contentions = []

    def access_state():
        """Access current_project state repeatedly."""
        thread_id = threading.get_ident()
        for _ in range(100):
            access_log.append((thread_id, pm.current_project))
            time.sleep(0.0001)  # Small delay to increase chance of races

    def modify_state():
        """Modify state while other threads are reading."""
        thread_id = threading.get_ident()
        for _ in range(50):
            try:
                pm.switch_project("project_lock_test")
                access_log.append((thread_id, pm.current_project))
                time.sleep(0.0001)
            except Exception as e:
                lock_contentions.append(str(e))

    # Start reader and writer threads
    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=access_state))

    for _ in range(2):
        threads.append(threading.Thread(target=modify_state))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=15.0)

    # Verify operations completed without corruption
    assert len(lock_contentions) == 0, f"Lock contentions detected: {lock_contentions}"

    # Verify all logged states are valid (should be project_lock_test after first switch)
    # Note: We may see initial state before first switch completes
    valid_states = {"project_lock_test", settings.default_project}
    for thread_id, state in access_log:
        assert state in valid_states, f"Invalid state detected: {state}"

    # Verify final state
    assert pm.current_project == "project_lock_test"


def test_choose_project_thread_safe(temp_storage_settings, caplog):
    """Test Bug H1: choose_project() reads current_project without lock.

    Verifies that choose_project() captures current_project atomically
    and doesn't see stale values during concurrent switches.
    """
    import logging
    storage = temp_storage_settings

    # Create multiple projects
    projects = ["alpha", "beta", "gamma"]
    _create_project_directories(storage, projects)

    pm = get_project_manager()
    results = []
    errors = []

    def score_and_choose():
        """Call choose_project while others switch projects."""
        try:
            for i in range(20):
                result = pm.choose_project("test query about python")
                # Verify current_project in result is one of our projects
                if "current_project" in result:
                    current = result["current_project"]
                    assert current in projects + [settings.default_project], \
                        f"Invalid current_project: {current}"
                results.append(result)
        except Exception as e:
            errors.append(str(e))

    def switch_projects():
        """Switch between projects concurrently."""
        try:
            for i in range(20):
                target = projects[i % len(projects)]
                pm.switch_project(target)
        except Exception as e:
            errors.append(str(e))

    # Run concurrent operations
    t1 = threading.Thread(target=score_and_choose)
    t2 = threading.Thread(target=switch_projects)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Check no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Check operations completed
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"

    # Verify all results have valid current_project values
    for result in results:
        assert "current_project" in result
        assert result["current_project"] in projects + [settings.default_project]


def test_discover_projects_concurrent(temp_storage_settings):
    """Test Bug H2: discover_projects() not thread-safe.

    Verifies that concurrent discovery operations return consistent
    project lists and don't miss projects or include partial deletions.
    """
    storage = temp_storage_settings

    # Start with some projects
    initial_projects = ["existing_1", "existing_2"]
    _create_project_directories(storage, initial_projects)

    pm = get_project_manager()
    discovery_results = []
    creation_results = []

    def discover_repeatedly():
        """Discover projects multiple times."""
        for i in range(50):
            projects = pm.discover_projects()
            discovery_results.append(projects)

    def create_concurrently():
        """Create new projects while discoveries run."""
        for i in range(3, 7):
            project_name = f"project_{i}"
            result = pm.create_project(project_name)
            creation_results.append((project_name, result))

    # Run concurrent operations
    t1 = threading.Thread(target=discover_repeatedly)
    t2 = threading.Thread(target=create_concurrently)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Verify all creations succeeded
    assert len(creation_results) == 4, f"Expected 4 creations, got {len(creation_results)}"
    for name, result in creation_results:
        assert "success" in result, f"Failed to create {name}: {result}"

    # Verify discoveries are consistent (monotonic increase is acceptable)
    # Each discovery should return a valid set of projects
    for projects in discovery_results:
        # All projects should be valid project names
        for p in projects:
            assert p in initial_projects + [f"project_{i}" for i in range(3, 7)] + \
                [settings.default_project], f"Invalid project name: {p}"

    # Final discovery should include all projects
    final_projects = pm.discover_projects()
    expected_projects = initial_projects + [f"project_{i}" for i in range(3, 7)]
    for expected in expected_projects:
        assert expected in final_projects, f"Project {expected} missing from final discovery"


def test_current_project_property_thread_safe(temp_storage_settings):
    """Test Bug H5: current_project property has no lock.

    Verifies that reading current_project property returns consistent
    values during concurrent switches.
    """
    storage = temp_storage_settings

    # Create projects
    projects = ["proj_x", "proj_y", "proj_z"]
    _create_project_directories(storage, projects)

    # Create fresh PM instance for this test
    pm = ProjectManager()
    read_values = []
    switch_count = {"count": 0}
    errors = []

    def read_property():
        """Read current_project property repeatedly."""
        try:
            for _ in range(50):
                value = pm.current_project
                read_values.append(value)
                # Verify it's always a valid project
                assert value in projects + [settings.default_project], \
                    f"Invalid project value: {value}"
        except Exception as e:
            errors.append(("read", str(e)))

    def switch_property():
        """Switch projects concurrently."""
        try:
            for i in range(25):
                target = projects[i % len(projects)]
                pm.switch_project(target)
                switch_count["count"] += 1
        except Exception as e:
            errors.append(("switch", str(e)))

    # Run concurrent operations
    threads = [
        threading.Thread(target=read_property),
        threading.Thread(target=read_property),
        threading.Thread(target=switch_property),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    # Check no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify operations completed
    assert len(read_values) == 100, f"Expected 100 reads, got {len(read_values)}"
    assert switch_count["count"] == 25, f"Expected 25 switches, got {switch_count['count']}"

    # Verify all reads were valid
    for value in read_values:
        assert value in projects + [settings.default_project]


def test_infer_project_logs_empty(temp_storage_settings, caplog):
    """Test Bug H4: infer_project() doesn't log when returning None.

    Verifies that infer_project() logs a debug message when
    discover_projects() returns an empty list.
    """
    import logging

    storage = temp_storage_settings

    pm = get_project_manager()

    # Clear any existing logs
    caplog.clear()

    # Enable debug logging
    with caplog.at_level(logging.DEBUG):
        # Try to infer when no projects exist
        result = pm.infer_project("test query")

        # Should return None
        assert result is None, "Expected None when no projects exist"

        # Should log a debug message
        assert len(caplog.records) > 0, "Expected debug log when no projects found"

        # Find the specific log message
        debug_logs = [record for record in caplog.records
                     if record.levelno == logging.DEBUG and
                     "infer_project" in record.message and
                     "No projects discovered" in record.message]

        assert len(debug_logs) > 0, \
            f"Expected 'infer_project: No projects discovered' log, got: {[r.message for r in caplog.records]}"


def test_infer_project_with_projects(temp_storage_settings):
    """Test infer_project() works correctly when projects exist."""
    storage = temp_storage_settings

    # Create projects with metadata
    _create_project_directories(storage, ["python_project", "javascript_project"])

    pm = get_project_manager()

    # Test basic inference - project name matching should work
    # The scoring system matches project names in the text
    result = pm.infer_project("help with python_project")
    # Should return a result (not None)
    assert result is not None, "Expected a project to be inferred"
    assert result in ["python_project", "javascript_project"], \
        f"Expected valid project, got {result}"

    result = pm.infer_project("help with javascript_project")
    assert result is not None, "Expected a project to be inferred"
    assert result in ["python_project", "javascript_project"], \
        f"Expected valid project, got {result}"


def test_choose_project_with_current_boost(temp_storage_settings):
    """Test that choose_project() correctly boosts current project score."""
    storage = temp_storage_settings

    # Create projects
    _create_project_directories(storage, ["current_proj", "other_proj"])

    pm = get_project_manager()

    # Switch to current_proj
    pm.switch_project("current_proj")

    # Query that might match both projects
    result = pm.choose_project("help with code")

    assert "candidates" in result
    assert len(result["candidates"]) == 2

    # Find current_proj in candidates
    current_candidate = next(c for c in result["candidates"] if c["project"] == "current_proj")
    other_candidate = next(c for c in result["candidates"] if c["project"] == "other_proj")

    # current_proj should have higher score due to current workspace boost
    assert current_candidate["score"] > other_candidate["score"], \
        f"Current project should have higher score: {current_candidate['score']} vs {other_candidate['score']}"

    # Should have "current workspace" in reasons
    assert "current workspace" in current_candidate["reasons"]


def test_directives_with_dashes(temp_storage_settings):
    """Test Bug M11: _parse_directives handles dashed project names.

    Verifies that project names with dashes (e.g., "my-project") are
    correctly preserved in the normalized text and can be matched
    in inclusion/exclusion directives.
    """
    storage = temp_storage_settings

    # Create projects with dashes and underscores
    projects = ["my-project", "test_project", "another-dash-project"]
    _create_project_directories(storage, projects)

    pm = get_project_manager()

    # Test inclusion directive with dashed project name
    result = pm.choose_project("focus only on my-project for python code")
    assert "error" not in result
    assert "candidates" in result

    # The dashed project should be included (not filtered out)
    candidate_names = [c["project"] for c in result["candidates"]]
    assert "my-project" in candidate_names, "Dashed project should be in candidates"

    # Test that the project name appears in the normalized text correctly
    includes, excludes = pm._parse_directives("focus only on my-project", projects)
    assert "my-project" in includes, "Dashed project should be detected in inclusion directive"


def test_directives_with_underscores(temp_storage_settings):
    """Test Bug M11: _parse_directives handles underscored project names."""
    storage = temp_storage_settings

    # Create projects with underscores
    projects = ["test_project", "my_awesome_project"]
    _create_project_directories(storage, projects)

    pm = get_project_manager()

    # Test inclusion directive with underscored project name
    result = pm.choose_project("specifically focus on test_project for help")
    assert "error" not in result

    includes, excludes = pm._parse_directives("specifically focus on test_project", projects)
    assert "test_project" in includes, "Underscored project should be detected"


def test_directives_exclusion_with_dashes(temp_storage_settings):
    """Test Bug M11: Exclusion directives work with dashed project names."""
    storage = temp_storage_settings

    projects = ["my-project", "other-project", "third_project"]
    _create_project_directories(storage, projects)

    pm = get_project_manager()

    # Test exclusion directive with dashed project name
    result = pm.choose_project("ignore my-project and focus on others")
    assert "error" not in result

    includes, excludes = pm._parse_directives("ignore my-project from search", projects)
    assert "my-project" in excludes, "Dashed project should be detected in exclusion directive"


def test_list_projects_current_consistent(temp_storage_settings):
    """Test Bug M12: list_projects() returns accurate current_project.

    Verifies that list_projects() uses a lock when reading _current_project,
    preventing race conditions where stale values could be reported.
    """
    storage = temp_storage_settings

    # Create projects
    _create_project_directories(storage, ["project_a", "project_b"])

    pm = get_project_manager()

    # Switch to project_a
    pm.switch_project("project_a")

    # Verify list_projects reports correct current project
    result = pm.list_projects()
    assert result["current_project"] == "project_a", \
        f"Expected current_project='project_a', got '{result['current_project']}'"

    # Switch to project_b
    pm.switch_project("project_b")

    # Verify list_projects reports updated current project
    result = pm.list_projects()
    assert result["current_project"] == "project_b", \
        f"Expected current_project='project_b', got '{result['current_project']}'"


def test_list_projects_current_concurrent(temp_storage_settings):
    """Test Bug M12: list_projects() is thread-safe for concurrent access."""
    storage = temp_storage_settings

    # Create projects
    _create_project_directories(storage, ["proj_1", "proj_2", "proj_3"])

    pm = get_project_manager()
    pm.switch_project("proj_1")

    results = []
    errors = []

    def list_and_verify():
        """List projects and verify current_project consistency."""
        try:
            for _ in range(20):
                result = pm.list_projects()
                # Verify current_project is always valid
                current = result["current_project"]
                assert current in ["proj_1", "proj_2", "proj_3", settings.default_project], \
                    f"Invalid current_project: {current}"
                results.append(current)
        except Exception as e:
            errors.append(str(e))

    def switch_projects():
        """Switch projects while listings are happening."""
        try:
            for i in range(10):
                target = f"proj_{(i % 3) + 1}"
                pm.switch_project(target)
        except Exception as e:
            errors.append(str(e))

    # Run concurrent operations
    t1 = threading.Thread(target=list_and_verify)
    t2 = threading.Thread(target=switch_projects)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Check no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all results were valid
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    for current in results:
        assert current in ["proj_1", "proj_2", "proj_3", settings.default_project]


def test_max_candidates_validation(temp_storage_settings):
    """Test Bug M13: choose_project validates max_candidates parameter.

    Verifies that invalid max_candidates values (negative, non-integer)
    are rejected with appropriate error messages.
    """
    storage = temp_storage_settings

    # Create projects
    _create_project_directories(storage, ["proj_a", "proj_b", "proj_c"])

    pm = get_project_manager()

    # Test negative value
    result = pm.choose_project("test query", max_candidates=-5)
    assert "error" in result
    assert result["error"] == "invalid_max_candidates"
    assert "must be >=" in result["message"]

    # Test zero
    result = pm.choose_project("test query", max_candidates=0)
    assert "error" in result
    assert result["error"] == "invalid_max_candidates"

    # Test non-integer string
    result = pm.choose_project("test query", max_candidates="invalid")
    assert "error" in result
    assert result["error"] == "invalid_max_candidates"
    assert "valid integer" in result["message"]

    # Test None
    result = pm.choose_project("test query", max_candidates=None)
    assert "error" in result
    assert result["error"] == "invalid_max_candidates"

    # Test valid value at boundary
    result = pm.choose_project("test query", max_candidates=1)
    assert "error" not in result
    assert "candidates" in result
    assert len(result["candidates"]) == 1

    # Test value larger than available projects
    result = pm.choose_project("test query", max_candidates=100)
    assert "error" not in result
    assert len(result["candidates"]) == 3  # Only 3 projects available

    # Test normal valid value
    result = pm.choose_project("test query", max_candidates=2)
    assert "error" not in result
    assert len(result["candidates"]) == 2
