# MCP Control Room UI

A FastAPI bridge plus a Vue 3 UI to visualize MCP tools, recent requests, logs, and stats.

## Run HTTP API + UI (dev backend)
1. Install Python deps (FastAPI, uvicorn). If not yet installed:
   ```bash
   pip install fastapi uvicorn
   ```
2. Start the API server (serves /api/mcp/* and static UI if built):
   ```bash
   python api_server.py
   ```

## Frontend (Vue 3 + Vite)
Location: `frontend/`

Install and run:
```bash
cd frontend
npm install
npm run dev   # dev server at http://localhost:5173 (proxies /api to :8000)
```

Build for production:
```bash
npm run build
```
The static assets output to `frontend/dist`. `api_server.py` will serve them automatically at `/` when `dist` exists.

## API surface
- `GET /api/mcp/tools`     → list of MCP tools (name, type, desc)
- `GET /api/mcp/requests`  → recent request samples (status, route, latency, tool, user)
- `GET /api/mcp/logs`      → recent log lines
- `GET /api/mcp/stats`     → uptime_seconds, rpm (per last 60s), errors, index stats

## Notes
- This HTTP server is separate from the MCP stdio server (`mcp_server.py`); it reuses the same rag modules.
- Logging buffer is in-memory only (ring buffer).
- Request buffer is populated by the FastAPI middleware on /api/* calls.
