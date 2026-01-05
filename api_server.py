#!/usr/bin/env python3
"""FastAPI HTTP server to expose MCP telemetry and serve the Vue UI.

Endpoints (all under /api/mcp):
- GET /api/mcp/tools     -> list of tools with name/type/desc
- GET /api/mcp/requests  -> recent request samples (status, path, latency, tool, user)
- GET /api/mcp/logs      -> recent log lines
- GET /api/mcp/stats     -> uptime, rpm, errors, index stats

Static frontend: serve built assets from frontend/dist (Vite build output).

Run:
    uvicorn api_server:app --reload --port 8000

Note: This does not replace the MCP stdio server (mcp_server.py); it provides an HTTP
surface for the UI and lightweight telemetry.
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.config import setup_logging, settings, validate_required_keys
from rag.storage.index import get_index_manager
from rag.storage.chroma import get_chroma_manager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
LOG = setup_logging()

# Validate API keys at startup
try:
    validate_required_keys()
    LOG.info("API key validation passed")
except ValueError as e:
    LOG.error(f"API key validation failed: {e}")
    raise

# ---------------------------------------------------------------------------
# Telemetry buffers
# ---------------------------------------------------------------------------
class RingLogHandler(logging.Handler):
    def __init__(self, maxlen: int = settings.log_buffer_size):
        super().__init__()
        self.buffer: Deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self.buffer.appendleft(msg)
        except Exception:  # pragma: no cover - safety net
            pass

    def dump(self) -> List[str]:
        with self._lock:
            return list(self.buffer)


log_handler = RingLogHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(log_handler)
# Preserve whichever level setup_logging configured; avoid redundant setLevel

request_buffer_lock = threading.Lock()
request_buffer: Deque[Dict[str, Any]] = deque(maxlen=settings.request_buffer_size)
process_start = time.time()

# Tool metadata (static list to avoid MCP context requirements)
TOOLS: List[Dict[str, str]] = [
    {"name": "query_rag", "type": "actions", "desc": "Semantic/hybrid search with auto-routing"},
    {"name": "index_documents", "type": "actions", "desc": "Index documents from a directory"},
    {"name": "add_text", "type": "actions", "desc": "Add raw text to index"},
    {"name": "inspect_directory", "type": "info", "desc": "Analyze directory before indexing"},
    {"name": "crawl_website", "type": "actions", "desc": "Crawl and index a website"},
    {"name": "manage_project", "type": "actions", "desc": "List/create/switch/update/analyze projects"},
    {"name": "get_stats", "type": "info", "desc": "Get system statistics"},
    {"name": "list_documents", "type": "info", "desc": "List indexed documents"},
    {"name": "clear_index", "type": "actions", "desc": "Clear the index"},
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MCP HTTP Bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_requests(request: Request, call_next):
    start = time.perf_counter()
    response: Optional[Response] = None
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        LOG.error(f"HTTP request failed: {e}", exc_info=True, extra={"path": str(request.url.path), "method": request.method})
        response = None
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code if response else 500
        entry = {
            "status": "ok" if status < 400 else "error",
            "route": str(request.url.path),
            "latency": round(duration_ms, 1),
            "tool": request.query_params.get("tool") or "-",
            "user": request.headers.get("x-user", "web"),
            "ts": time.time(),
            "code": status,
        }
        with request_buffer_lock:
            request_buffer.appendleft(entry)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cached_tools():
    return TOOLS


@app.get("/api/mcp/tools")
async def list_tools():
    return JSONResponse(_cached_tools())


@app.get("/api/mcp/requests")
async def list_requests():
    with request_buffer_lock:
        return JSONResponse(list(request_buffer))


@app.get("/api/mcp/logs")
async def list_logs():
    return JSONResponse(log_handler.dump())


@app.get("/api/mcp/health")
async def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - process_start, 1)}


@app.get("/api/mcp/stats")
async def stats():
    now = time.time()
    rpm = 0
    errors = 0
    with request_buffer_lock:
        buffer_copy = list(request_buffer)

    for r in buffer_copy:
        if now - r.get("ts", now) <= 60:
            rpm += 1
            if r.get("code", 0) >= 400:
                errors += 1
    uptime = now - process_start
    try:
        index_stats = {
            "status": "ready" if settings.storage_path.exists() else "missing",
            "documents": get_chroma_manager().get_collection_count(),
            "current_project": get_index_manager().current_project,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
        }
    except Exception as e:  # pragma: no cover - defensive
        LOG.warning("index stats failed: %s", e)
        index_stats = {"error": str(e)}
    payload = {
        "uptime_seconds": round(uptime, 1),
        "rpm": rpm,
        "errors": errors,
        "index": index_stats,
    }
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Static files (frontend build)
# ---------------------------------------------------------------------------
index_file = FRONTEND_DIST / "index.html"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
async def serve_index():
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "frontend not built"}, status_code=404)


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # SPA fallback
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "frontend not built"}, status_code=404)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
