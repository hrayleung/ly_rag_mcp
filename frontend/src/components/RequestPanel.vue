<template>
  <section class="panel request-panel">
    <div class="panel-header">
      <div class="header-title">LIVE REQUESTS</div>
      <div v-if="requests.length" class="pulse-indicator"></div>
    </div>

    <div class="table-container">
      <table class="request-table">
        <thead>
          <tr>
            <th class="col-status">STATUS</th>
            <th class="col-method">METHOD</th>
            <th class="col-route">ROUTE</th>
            <th class="col-tool">TOOL</th>
            <th class="col-latency align-right">LATENCY</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in requests" :key="i" class="request-row">
            <td class="col-status">
              <span class="status-box" :class="r.status || 'ok'">
                {{ (r.status === 'error' ? 'ERR' : 'OK') }}
              </span>
            </td>
            <td class="col-method">
              <span class="method-text mono">POST</span>
            </td>
            <td class="col-route">
              <span class="route-text mono">{{ r.route }}</span>
            </td>
            <td class="col-tool">
              <span class="tool-text">{{ r.tool || '—' }}</span>
            </td>
            <td class="col-latency align-right">
              <span class="latency-val mono">{{ r.latency }}</span><span class="unit">ms</span>
            </td>
          </tr>
          <tr v-if="!requests.length" class="empty-row">
            <td colspan="5" class="empty-state">
              <span>Waiting for requests...</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { RequestEntry } from '@/composables/useMcpApi'
defineProps<{ requests: RequestEntry[] }>()
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.panel-header {
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
}

.header-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--fg-muted);
}

.pulse-indicator {
  width: 4px;
  height: 4px;
  background: var(--success);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--success);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 0.2; }
  50% { opacity: 1; }
  100% { opacity: 0.2; }
}

.table-container {
  flex: 1;
  overflow-y: auto;
  background: var(--bg-panel); /* Slightly distinct from header */
}

/* Table Layout */
.request-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed; /* Strict layout */
}

.request-table th {
  position: sticky;
  top: 0;
  text-align: left;
  padding: 8px 16px;
  font-size: 10px;
  font-weight: 600;
  color: var(--fg-muted);
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  z-index: 1;
}

.request-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  color: var(--fg-dim);
  height: 48px; /* Fixed row height */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.request-row:hover td {
  background: var(--bg-hover);
  color: var(--fg);
}

/* Column Widths */
.col-status { width: 60px; }
.col-method { width: 80px; }
.col-route { width: 30%; }
.col-tool { width: auto; }
.col-latency { width: 100px; }

/* Cell Styling */
.status-box {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
}
.status-box.ok { color: var(--success); }
.status-box.error { color: var(--error); }

.method-text {
  font-size: 11px;
  opacity: 0.7;
}

.route-text {
  color: var(--fg);
}

.tool-text {
  color: var(--fg-muted);
}

.latency-val {
  color: var(--fg);
}

.unit {
  font-size: 10px;
  color: var(--fg-muted);
  margin-left: 2px;
}

.align-right {
  text-align: right;
}

.empty-state {
  text-align: center;
  padding: 64px 0;
  color: var(--fg-muted);
  font-size: 13px;
  border-bottom: none;
}
</style>
