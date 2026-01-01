<template>
  <section class="panel request-panel">
    <div class="panel-header">
      <h2>Live Requests</h2>
      <div v-if="requests.length" class="pulse-indicator"></div>
    </div>
    <div class="panel-content">
      <table class="request-table" aria-label="Live request table">
        <thead>
          <tr>
            <th width="80">Status</th>
            <th>Method / Route</th>
            <th>Tool</th>
            <th class="align-right">Duration</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in requests" :key="i" class="request-row">
            <td>
              <span class="status-dot" :class="r.status || 'ok'"></span>
            </td>
            <td>
              <div class="route-info">
                <span class="method">POST</span>
                <span class="route">{{ r.route }}</span>
              </div>
            </td>
            <td class="tool-cell">{{ r.tool || '—' }}</td>
            <td class="latency-cell align-right">
              {{ r.latency }}<span class="unit">ms</span>
            </td>
          </tr>
          <tr v-if="!requests.length" class="empty-row">
            <td colspan="4" class="empty-state">
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
defineProps<{ requests: RequestEntry[]; outlined?: boolean }>()
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  animation: fadeIn 0.3s var(--ease);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 4px;
}

h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
}

.pulse-indicator {
  width: 6px;
  height: 6px;
  background: var(--success);
  border-radius: 50%;
  animation: pulse-opacity 2s infinite;
}

@keyframes pulse-opacity {
  0% { opacity: 0.4; }
  50% { opacity: 1; }
  100% { opacity: 0.4; }
}

.panel-content {
  flex: 1;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--bg-warm);
}

.request-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 12px;
}

.request-table th {
  text-align: left;
  padding: 12px 16px;
  font-weight: 500;
  color: var(--muted);
  border-bottom: 1px solid var(--line);
  font-family: var(--font-body);
  font-size: 11px;
}

.request-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--line-dim);
  color: var(--text-dim);
}

.request-row:last-child td {
  border-bottom: none;
}

.request-row:hover {
  background: var(--panel-light);
}

/* Status */
.status-dot {
  display: block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--muted);
}
.status-dot.ok { background: var(--success); }
.status-dot.error { background: var(--error); }

/* Route Info */
.route-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.method {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  background: var(--bg);
  padding: 2px 4px;
  border-radius: 4px;
  border: 1px solid var(--line-dim);
}

.route {
  color: var(--text);
}

/* Cells */
.tool-cell {
  opacity: 0.8;
}

.latency-cell {
  font-feature-settings: "tnum";
}

.unit {
  color: var(--muted);
  font-size: 10px;
  margin-left: 2px;
}

.align-right {
  text-align: right;
}

.empty-state {
  text-align: center;
  padding: 48px 0;
  color: var(--muted);
  font-family: var(--font-body);
}
</style>
