"""Tests for project manager scoring and choose_project.

These tests create a temporary storage layout and minimal metadata files,
then assert choose_project returns expected rankings and directive behavior.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def temp_storage_settings(tmp_path, monkeypatch):
    """Point rag.settings.storage_path at a temp directory."""
    from rag.config import settings

    original_storage = settings.storage_path
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    yield tmp_path
    monkeypatch.setattr(settings, "storage_path", original_storage)


def _write_metadata(storage_root, project: str, payload: dict):
    proj_dir = storage_root / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    meta_path = proj_dir / "project_metadata.json"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def test_choose_project_ranks_by_keywords(temp_storage_settings):
    storage = temp_storage_settings

    _write_metadata(
        storage,
        "frontend",
        {
            "name": "frontend",
            "display_name": "Frontend",
            "keywords": ["react", "vite"],
            "default_paths": ["/repo/frontend"],
            "description": "",
        },
    )
    _write_metadata(
        storage,
        "backend",
        {
            "name": "backend",
            "display_name": "Backend",
            "keywords": ["fastapi", "database"],
            "default_paths": ["/repo/backend"],
            "description": "",
        },
    )

    from rag.project.manager import ProjectManager

    pm = ProjectManager()
    pm._current_project = "backend"

    resp = pm.choose_project("How does react routing work?", max_candidates=2)

    assert resp["recommendation"] == "frontend"
    assert [c["project"] for c in resp["candidates"]] == ["frontend", "backend"]


def test_choose_project_parses_includes_and_excludes(temp_storage_settings):
    storage = temp_storage_settings

    _write_metadata(
        storage,
        "frontend",
        {"name": "frontend", "keywords": ["react"], "default_paths": []},
    )
    _write_metadata(
        storage,
        "backend",
        {"name": "backend", "keywords": ["api"], "default_paths": []},
    )

    from rag.project.manager import ProjectManager

    pm = ProjectManager()

    only_backend = pm.choose_project("Only backend: how does auth work?", max_candidates=3)
    assert only_backend["includes"] == ["backend"]
    assert all(c["project"] == "backend" for c in only_backend["candidates"])

    exclude_frontend = pm.choose_project("Investigate api errors, excluding frontend", max_candidates=3)
    assert "frontend" in exclude_frontend["excludes"]
    assert all(c["project"] != "frontend" for c in exclude_frontend["candidates"])


def test_choose_project_errors_on_missing_question(temp_storage_settings):
    # Ensure there is at least one project so we don't hit no_projects first.
    _write_metadata(temp_storage_settings, "p", {"name": "p"})

    from rag.project.manager import ProjectManager

    pm = ProjectManager()

    assert pm.choose_project("") == {"error": "question_required"}
    assert pm.choose_project("   \n") == {"error": "question_required"}
