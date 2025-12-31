import time
import logging

import pytest
from fastapi.testclient import TestClient

import api_server


def _stub_index_manager(current_project="demo"):
    class _Mgr:
        def __init__(self):
            self.current_project = current_project
    return _Mgr()


def _stub_chroma_manager(count=3):
    class _Mgr:
        def get_collection_count(self):
            return count
    return _Mgr()


@pytest.fixture(autouse=True)
def clear_buffers(monkeypatch, tmp_path):
    api_server.request_buffer.clear()
    api_server.log_handler.buffer.clear()
    # Stub managers to avoid hitting real storage
    monkeypatch.setattr(api_server, "get_index_manager", lambda: _stub_index_manager())
    monkeypatch.setattr(api_server, "get_chroma_manager", lambda: _stub_chroma_manager())
    # Point storage_path to a temp dir
    monkeypatch.setattr(api_server.settings, "storage_path", tmp_path)
    return None


@pytest.fixture()
def client():
    return TestClient(api_server.app)


def test_health(client):
    resp = client.get("/api/mcp/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0


def test_tools_list(client):
    resp = client.get("/api/mcp/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(t["name"] == "query_rag" for t in data)


def test_stats_counts_rpm_and_errors(client):
    now = time.time()
    api_server.request_buffer.appendleft({"ts": now, "code": 200})
    api_server.request_buffer.appendleft({"ts": now, "code": 500})

    resp = client.get("/api/mcp/stats")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["rpm"] == 2
    assert payload["errors"] == 1
    assert payload["index"]["documents"] == 3
    assert payload["index"]["current_project"] == "demo"


def test_logs_endpoint_returns_ring_buffer(client):
    logging.getLogger(__name__).info("hello from test")
    resp = client.get("/api/mcp/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_frontend_not_built_returns_404(client):
    resp = client.get("/")
    assert resp.status_code == 404
    assert resp.json().get("error") == "frontend not built"
