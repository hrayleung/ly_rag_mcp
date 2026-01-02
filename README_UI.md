# MCP Control Room UI

A Vue 3 + Vite single-page application for monitoring and controlling the RAG MCP server. Displays live requests, available tools, server logs, and system statistics.

## Architecture

### Tech Stack
- **Framework**: Vue 3 (v3.4.21) with Composition API and `<script setup>` syntax
- **Router**: Vue Router (v4.6.4)
- **Build Tool**: Vite (v5.4.21) with TypeScript support
- **Plugin**: `@vitejs/plugin-vue` for Vue SFC support

### Component Structure
```
frontend/src/
├── App.vue                 # Root component
├── main.ts                 # Entry point
├── style.css               # Global styles + font imports
├── router/
│   └── index.ts           # Single route configuration
├── components/
│   ├── HeaderBar.vue      # App header with title and latency badge
│   ├── StatsSidebar.vue   # Navigation, actions, stats cards
│   ├── RequestPanel.vue   # Live request monitoring table
│   ├── ToolPanel.vue      # Tool listing with filtering
│   └── LogPanel.vue       # Server event log panel
├── composables/
│   ├── useMcpApi.ts       # API client composable
│   └── usePolling.ts      # Auto-polling with latency measurement
├── styles/
│   └── variables.css      # CSS custom properties (design system)
└── views/
    └── McpControlRoom.vue # Main dashboard view
```

### Main View Layout (McpControlRoom.vue)
- **Three main panels**: Live Requests, Tools, Logs
- **Stats sidebar** with navigation buttons and metrics cards:
  - Uptime
  - Requests per minute
  - Error count
  - Active tools

### Design System
- Dark theme with green accent (`#b4ff3f`) and orange secondary (`#ff9f1c`)
- Google Fonts: Saira Semi Condensed (UI) and IBM Plex Mono (code)
- CSS Grid responsive layout
- CSS custom properties in `variables.css`

## Run HTTP API + UI (dev backend)

### 1. Install Python deps
```bash
pip install fastapi uvicorn
```

### 2. Start API server
```bash
python api_server.py
```
Server runs on `http://0.0.0.0:8000` by default.

### 3. Frontend (Optional Development Mode)

Install and run dev server:
```bash
cd frontend
npm install
npm run dev   # dev server at http://localhost:5173
```

**Vite Proxy Configuration**: The dev server proxies `/api` requests to `localhost:8000` backend.

### 4. Build for Production

```bash
cd frontend
npm run build
```

The static assets output to `frontend/dist/`. `api_server.py` will serve them automatically at `/` when the `dist` directory exists.

### 5. Access the UI

Once `api_server.py` is running:
- Production build: Visit `http://localhost:8000`
- Dev mode: Visit `http://localhost:5173`

## API Surface

All API endpoints are prefixed with `/api/mcp/`:

| Endpoint | Description | Response Format |
|----------|-------------|-----------------|
| `GET /tools` | List registered tools | `{ name, type, description }[]` |
| `GET /requests` | Recent request samples | `{ status, route, latency, tool, user }[]` |
| `GET /logs` | Server log lines | Ring buffer of log text |
| `GET /stats` | System metrics | `{ uptime, rpm, errors, index_stats }` |
| `GET /health` | Health check | Uptime status |

### API Client (useMcpApi.ts)

The `useMcpApi` composable fetches data in parallel every 3 seconds:
```typescript
const { tools, requests, logs, stats, error, loading } = useMcpApi();
```

Data is fetched via `Promise.all()` on component mount.

### Polling Composable (usePolling.ts)

Auto-polling with latency measurement:
- Default interval: 3 seconds
- Measures request latency
- Handles errors gracefully

## Notes

- This HTTP server is **separate** from the MCP stdio server (`mcp_server.py`); it reuses the same `rag` modules
- Logging buffer is **in-memory only** (ring buffer)
- Request buffer is populated by FastAPI middleware only on `/api/*` calls
- The UI provides real-time monitoring but does not modify MCP server state
- All MCP tool execution still happens through stdio protocol
