<template>
  <aside class="sidebar">
    <div class="section-title">Navigation</div>
    <div class="nav-list">
      <button
        class="nav-item"
        :class="{ active: activePanel === 'requests' }"
        @click="$emit('panel', 'requests')"
      >
        <span class="nav-label">Requests</span>
        <span class="nav-count" v-if="stats.rpm">{{ stats.rpm }}</span>
      </button>
      <button
        class="nav-item"
        :class="{ active: activePanel === 'tools' }"
        @click="$emit('panel', 'tools')"
      >
        <span class="nav-label">Tools</span>
        <span class="nav-count">{{ toolsLength }}</span>
      </button>
      <button
        class="nav-item"
        :class="{ active: activePanel === 'logs' }"
        @click="$emit('panel', 'logs')"
      >
        <span class="nav-label">Logs</span>
      </button>
    </div>

    <!-- Spacer -->
    <div style="flex: 1"></div>

    <div class="section-title">Controls</div>
    <div class="action-list">
      <button class="action-btn" @click="$emit('refresh')">
        Refresh Data
      </button>
      <button class="action-btn" @click="$emit('simulate')">
        Test Trigger
      </button>
      <button class="action-btn danger" @click="$emit('clear-log')">
        Clear Logs
      </button>
    </div>

    <div class="section-title">System Status</div>
    <div class="stats-minimal">
      <div class="stat-row">
        <span>Uptime</span>
        <span class="mono">{{ stats.uptime || '—' }}</span>
      </div>
      <div class="stat-row">
        <span>Errors</span>
        <span class="mono" :class="{ 'has-errors': stats.errors }">{{ stats.errors ?? 0 }}</span>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
interface StatsVm { uptime?: string; rpm?: number | null; errors?: number | null }
const props = defineProps<{ stats: StatsVm; toolsLength: number; activePanel: string }>()
</script>

<style scoped>
.sidebar {
  grid-area: sidebar;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 12px 4px;
  animation: slideInRight 0.4s var(--ease);
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  padding-left: 12px;
  margin-bottom: -12px; /* Pull closer to content */
}

/* Nav Items - Minimal "Folder" style */
.nav-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 8px 12px;
  border-radius: var(--radius);
  color: var(--text-dim);
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.nav-item:hover {
  color: var(--text);
  background: var(--panel-light);
}

.nav-item.active {
  color: var(--text);
  background: var(--bg-elevated);
  box-shadow: 0 0 0 1px var(--line);
}

.nav-count {
  font-size: 11px;
  background: var(--bg);
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--muted);
  font-family: var(--font-mono);
  border: 1px solid var(--line-dim);
}

/* Actions */
.action-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.action-btn {
  width: 100%;
  text-align: left;
  padding: 8px 12px;
  font-size: 13px;
  color: var(--text-dim);
  border: 1px solid transparent;
  border-radius: var(--radius);
  transition: all 0.2s ease;
}

.action-btn:hover {
  border-color: var(--line);
  background: var(--panel-light);
  color: var(--text);
}

.action-btn.danger:hover {
  color: var(--error);
  border-color: var(--error-dim);
  background: rgba(239, 68, 68, 0.05); /* Error tint */
}

/* Minimal Stats */
.stats-minimal {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 12px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--muted);
}

.mono {
  font-family: var(--font-mono);
  color: var(--text-dim);
}

.has-errors {
  color: var(--error);
}
</style>
