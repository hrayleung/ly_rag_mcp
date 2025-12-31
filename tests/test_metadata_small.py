"""Additional coverage for MetadataManager save/save_manifest fallbacks and caching."""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

import pytest

import rag.project.metadata as meta_mod


@pytest.fixture()
def tmp_project(tmp_path, monkeypatch):
    # ensure storage path points to tmp
    monkeypatch.setattr(meta_mod.settings, "storage_path", tmp_path)
    return tmp_path


def test_save_fallback_on_lock_error(tmp_project, monkeypatch):
    mm = meta_mod.MetadataManager()
    # force lock to raise
    monkeypatch.setattr(meta_mod, "_file_lock", lambda path: (_ for _ in ()).throw(OSError("boom")))
    data = meta_mod.ProjectMetadata(name="p")
    out = mm.save("p", data)
    meta_path = tmp_project / "p" / meta_mod.settings.project_metadata_filename
    assert meta_path.exists()
    saved = json.loads(meta_path.read_text())
    assert saved["name"] == "p"
    assert mm._cache["p"].name == "p"


def test_save_manifest_fallback(tmp_project, monkeypatch):
    mm = meta_mod.MetadataManager()
    monkeypatch.setattr(meta_mod, "_file_lock", lambda path: (_ for _ in ()).throw(OSError("fail")))
    mm.save_manifest("p", {"roots": {}})
    manifest_path = tmp_project / "p" / meta_mod.settings.ingest_manifest_filename
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["roots"] == {}
    assert "updated_at" in data


def test_load_uses_cache(tmp_project):
    mm = meta_mod.MetadataManager()
    meta_path = tmp_project / "p" / meta_mod.settings.project_metadata_filename
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"name": "p", "description": "d"}), encoding="utf-8")

    first = mm.load("p")
    second = mm.load("p")
    assert first is second
    assert first.description == "d"
