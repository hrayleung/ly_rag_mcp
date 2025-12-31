from pathlib import Path

from rag.config import settings
from rag.project.manager import ProjectManager


def test_discover_projects_requires_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_path", tmp_path)

    valid_with_metadata = tmp_path / "proj_meta"
    valid_with_metadata.mkdir()
    (valid_with_metadata / settings.project_metadata_filename).write_text("{}")

    valid_with_docstore = tmp_path / "proj_docstore"
    valid_with_docstore.mkdir()
    (valid_with_docstore / "docstore.json").write_text("{}")

    valid_with_chroma = tmp_path / "proj_chroma"
    valid_with_chroma.mkdir()
    (valid_with_chroma / "chroma_db").mkdir()

    invalid_empty = tmp_path / "empty"
    invalid_empty.mkdir()

    hidden = tmp_path / ".hidden"
    hidden.mkdir()

    chroma_root = tmp_path / "chroma_db"
    chroma_root.mkdir()

    manager = ProjectManager()
    projects = set(manager.discover_projects())

    assert "proj_meta" in projects
    assert "proj_docstore" in projects
    assert "proj_chroma" in projects

    assert "empty" not in projects
    assert ".hidden" not in projects
    assert "chroma_db" not in projects
