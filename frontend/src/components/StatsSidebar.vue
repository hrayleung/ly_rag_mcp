<template>
  <aside class="sidebar">
    <div class="nav-section">
      <div class="section-label">NAVIGATE</div>
      <div class="nav-list">
        <button
          class="nav-item"
          :class="{ active: activePanel === 'requests' }"
          @click="$emit('panel', 'requests')"
        >
          <span class="nav-text">Requests</span>
          <span class="nav-meta mono" v-if="stats.rpm">{{ stats.rpm }}</span>
        </button>
        <button
          class="nav-item"
          :class="{ active: activePanel === 'tools' }"
          @click="$emit('panel', 'tools')"
        >
          <span class="nav-text">Tools</span>
          <span class="nav-meta mono">{{ toolsLength }}</span>
        </button>
      </div>
    </div>

    <!-- Controls -->
    <div class="nav-section">
      <div class="section-label">ACTIONS</div>
      <div class="nav-list">
        <button class="nav-item action" @click="$emit('refresh')">Refresh</button>
        <button class="nav-item action" @click="$emit('simulate')">Test Trigger</button>
        <button class="nav-item action danger" @click="$emit('clear-log')">Clear Logs</button>
      </div>
    </div>

    <div class="spacer"></div>

    <!-- System Info -->
    <div class="system-section">
      <div class="stat-row">
        <span class="stat-label">UPTIME</span>
        <span class="stat-val mono">{{ stats.uptime || '—' }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">ERRORS</span>
        <span class="stat-val mono" :class="{ 'text-error': stats.errors }">{{ stats.errors ?? 0 }}</span>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
interface StatsVm { uptime?: string; rpm?: number | null; errors?: number | null }
defineProps<{ stats: StatsVm; toolsLength: number; activePanel: string }>()
</script>

<style scoped>
.sidebar {
  padding: 24px 16px;
  display: flex;
  flex-direction: column;
  gap: 32px;
  height: 100%;
}

.section-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--fg-muted);
  margin-bottom: 12px;
  padding-left: 8px;
}

.nav-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.nav-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 8px 12px; /* Increased horizontal padding */
  font-size: 13px;
  color: var(--fg-muted);
  border-radius: var(--radius);
  transition: all 0.1s ease;
  border-left: 2px solid transparent; /* Prepare for active border */
}

/* Hover State */
.nav-item:hover {
  color: var(--fg);
  background: var(--bg-hover);
}

/* Active State */
.nav-item.active {
  color: var(--fg);
  background: var(--bg-hover);
  border-left-color: var(--fg); /* Minimal active indicator */
}

.nav-meta {
  font-size: 11px;
  opacity: 0.5;
}

.action {
  justify-content: flex-start;
}

.action.danger:hover {
  color: var(--error);
  background: rgba(239, 68, 68, 0.05); /* Slight red tint */
}

.spacer {
  flex: 1;
}

/* System Stats */
.system-section {
  padding: 16px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-label {
  font-size: 10px;
  color: var(--fg-muted);
  font-weight: 500;
  letter-spacing: 0.05em;
}

.stat-val {
  font-size: 12px;
  color: var(--fg);
}

.text-error {
  color: var(--error);
}
</style>
