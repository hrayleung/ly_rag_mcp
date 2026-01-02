<template>
  <div class="app">
    <header class="header">
      <div>
        <h1>MCP Control Room</h1>
        <small>Monitor | Trigger tools | Inspect telemetry</small>
      </div>
      <div class="header-right">
        <span class="badge">Live</span>
        <span class="badge-pill">Latency <span>{{ latencyLabel }}</span></span>
      </div>
    </header>

    <aside class="sidebar">
      <div class="section-title">Navigation</div>
      <div class="nav">
        <button class="btn" @click="activePanel = 'requests'">Live Requests <span class="hint">Traffic, status, durations</span></button>
        <button class="btn" @click="activePanel = 'tools'">Tools <span class="hint">List & invoke</span></button>
        <button class="btn" @click="activePanel = 'logs'">Logs <span class="hint">Server events</span></button>
        <button class="btn ghost" @click="refresh">↻ Refresh</button>
      </div>

      <div class="section-title">Actions</div>
      <div class="actions">
        <button class="primary" @click="simulateTrigger">Trigger tool</button>
        <button class="ghost" @click="clearLog">Clear log</button>
      </div>

      <div class="section-title">Stats</div>
      <div class="card-grid">
        <div class="card"><div class="label">Uptime</div><div class="value">{{ stats.uptime || '—' }}</div></div>
        <div class="card"><div class="label">Req / min</div><div class="value">{{ stats.rpm ?? '—' }}</div></div>
        <div class="card"><div class="label">Errors</div><div class="value">{{ stats.errors ?? '—' }}</div></div>
        <div class="card"><div class="label">Active tools</div><div class="value">{{ tools.length }}</div></div>
      </div>
    </aside>

    <main class="main">
      <section class="panel" :class="{ outlined: activePanel === 'requests' }">
        <div class="toolbar">
          <h2>Live Requests</h2>
          <span class="chip">Streaming</span>
        </div>
        <table class="table" aria-label="Live request table">
          <thead>
            <tr><th>Status</th><th>Route</th><th>Latency</th><th>Tool</th><th>User</th></tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in requests" :key="i" class="row">
              <td>
                <span class="status-dot" :style="{ background: r.status === 'ok' ? 'var(--accent)' : 'var(--accent-2)' }"></span>
                {{ (r.status || '').toUpperCase() }}
              </td>
              <td>{{ r.route }}</td>
              <td>{{ r.latency }} ms</td>
              <td>{{ r.tool }}</td>
              <td>{{ r.user }}</td>
            </tr>
            <tr v-if="!requests.length"><td colspan="5" class="empty">No requests yet</td></tr>
          </tbody>
        </table>
      </section>

      <section class="panel" :class="{ outlined: activePanel === 'tools' }">
        <div class="toolbar">
          <h2>Tools</h2>
          <div class="tabs">
            <div class="tab" :class="{ active: toolFilter === 'all' }" @click="toolFilter = 'all'">All</div>
            <div class="tab" :class="{ active: toolFilter === 'actions' }" @click="toolFilter = 'actions'">Actions</div>
            <div class="tab" :class="{ active: toolFilter === 'info' }" @click="toolFilter = 'info'">Info</div>
          </div>
        </div>
        <div class="card-grid" aria-label="Tool list">
          <div class="card" v-for="t in filteredTools" :key="t.name">
            <div class="label">{{ t.type }}</div>
            <div class="value">{{ t.name }}</div>
            <small>{{ t.desc }}</small>
            <button class="primary" style="margin-top:6px;" @click="log(`Invoked ${t.name}`)">Invoke</button>
          </div>
          <div v-if="!filteredTools.length" class="empty">No tools</div>
        </div>
      </section>

      <section class="panel wide" :class="{ outlined: activePanel === 'logs' }">
        <div class="toolbar">
          <h2>Logs</h2>
          <span class="chip">Console</span>
        </div>
        <div class="log" role="log" aria-live="polite">
          <div v-for="(line, i) in logs" :key="i">{{ line }}</div>
          <div v-if="!logs.length" class="empty">No logs yet</div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const activePanel = ref('requests')
const toolFilter = ref('all')
const requests = ref([])
const tools = ref([])
const logs = ref([])
const stats = ref({ uptime: '', rpm: null, errors: null })
const latencyLabel = ref('—')
let timer = null

const filteredTools = computed(() =>
  toolFilter.value === 'all' ? tools.value : tools.value.filter(t => t.type === toolFilter.value)
)

async function fetchJSON(url) {
  const t0 = performance.now()
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  const data = await res.json()
  latencyLabel.value = `${Math.round(performance.now() - t0)} ms`
  return data
}

async function load() {
  try { requests.value = await fetchJSON('/api/mcp/requests') } catch (e) { console.warn(e) }
  try { tools.value = await fetchJSON('/api/mcp/tools') } catch (e) { console.warn(e) }
  try {
    const logData = await fetchJSON('/api/mcp/logs')
    logs.value = (Array.isArray(logData) ? logData : []).slice(-200).reverse()
  } catch (e) { console.warn(e) }
}

function refresh() { load() }
function clearLog() { logs.value = [] }
function simulateTrigger() { log('Triggered sample tool') }
function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`
  logs.value = [line, ...logs.value].slice(0, 200)
}

onMounted(() => {
  load()
  timer = setInterval(load, 3000) // every 3s
})

onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
:root {
  --bg: #0c0d10; --panel: #111319; --panel-strong: #161922; --line: #1f2330;
  --text: #e7ecf5; --muted: #8b93a7; --accent: #b4ff3f; --accent-2: #ff9f1c;
  --radius: 12px; --grid: linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px),
                       linear-gradient(180deg, rgba(255,255,255,0.02) 1px, transparent 1px);
}
* { box-sizing: border-box; }
.app { min-height: 100vh; background: var(--bg); color: var(--text);
  background-image: var(--grid); background-size: 48px 48px;
  padding: 24px; display: grid; gap: 18px;
  grid-template-columns: 320px 1fr; grid-template-rows: auto 1fr;
  grid-template-areas: "sidebar header" "sidebar main";
}
.header { grid-area: header; background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 16px 18px; display: flex; align-items: center;
  justify-content: space-between; box-shadow: 0 14px 40px rgba(0,0,0,0.45);
}
h1 { margin: 0; font-size: 18px; letter-spacing: 0.03em; text-transform: uppercase; }
.badge, .badge-pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
  border-radius: 999px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
}
.badge { background: rgba(180,255,63,0.12); color: var(--accent); }
.badge-pill { background: rgba(180,255,63,0.1); color: var(--accent); border: 1px solid rgba(180,255,63,0.3); }
.header-right { display: flex; gap: 12px; align-items: center; }
.sidebar { grid-area: sidebar; background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 16px; box-shadow: 0 14px 40px rgba(0,0,0,0.45);
  display: flex; flex-direction: column; gap: 12px;
}
.section-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 8px 0 4px; }
.nav { display: grid; gap: 8px; }
.btn { width: 100%; background: var(--panel-strong); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 14px; color: var(--text); text-align: left; font-weight: 600;
  transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
}
.btn:hover { transform: translateY(-2px); border-color: rgba(180,255,63,0.35); box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
.btn .hint { display: block; font-size: 12px; color: var(--muted); margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
.primary { background: var(--accent); color: #0b0c0f; border: none; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; padding: 10px 12px; border-radius: 10px; cursor: pointer; }
.primary:hover { box-shadow: 0 10px 24px rgba(180,255,63,0.3); }
.ghost { background: transparent; border: 1px dashed var(--line); color: var(--muted); padding: 10px 12px; border-radius: 10px; cursor: pointer; }
.main { grid-area: main; display: grid; grid-template-columns: 1.1fr 0.9fr; grid-template-rows: 260px 1fr; gap: 16px; align-items: stretch; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px;
  box-shadow: 0 14px 40px rgba(0,0,0,0.45); display: flex; flex-direction: column; gap: 10px; position: relative;
}
.panel.wide { grid-column: span 2; }
.panel.outlined { outline: 1px solid rgba(180,255,63,0.35); }
.toolbar { display: flex; align-items: center; gap: 10px; }
h2 { margin: 0; font-size: 14px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.chip { padding: 6px 10px; border-radius: 999px; background: rgba(255,159,28,0.12); color: var(--accent-2); font-size: 12px; border: 1px solid rgba(255,159,28,0.35); }
.table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; }
.table th, .table td { padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 13px; }
.table th { text-align: left; color: var(--muted); letter-spacing: 0.05em; }
.row { background: linear-gradient(90deg, rgba(180,255,63,0.03), transparent); }
.row:hover { background: linear-gradient(90deg, rgba(180,255,63,0.08), rgba(255,159,28,0.05)); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 12px rgba(180,255,63,0.5); }
.log { background: #0a0b0e; border: 1px solid #171a24; border-radius: 10px; padding: 10px; height: 180px; overflow: auto; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #d6dbeb; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel-strong); display: grid; gap: 4px; position: relative; overflow: hidden; }
.card:before { content: ""; position: absolute; inset: -20% 40%; width: 120%; height: 120%; background: radial-gradient(circle at 50% 50%, rgba(180,255,63,0.12), transparent 55%); opacity: 0; transition: opacity 220ms ease; }
.card:hover:before { opacity: 1; }
.label { color: var(--muted); font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }
.value { font-family: 'IBM Plex Mono', monospace; font-size: 15px; }
.tabs { display: flex; gap: 8px; }
.tab { padding: 8px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel-strong); font-size: 13px; cursor: pointer; transition: border-color 160ms ease, color 160ms ease; }
.tab.active { border-color: rgba(180,255,63,0.5); color: var(--accent); }
.empty { color: var(--muted); text-align: center; padding: 12px; font-size: 13px; }
@media (max-width: 1080px) {
  .app { grid-template-columns: 1fr; grid-template-areas: "header" "main"; }
  .sidebar { grid-area: header; order: 2; }
  .header { order: 1; }
  .main { grid-template-columns: 1fr; grid-template-rows: auto; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
}
</style>
