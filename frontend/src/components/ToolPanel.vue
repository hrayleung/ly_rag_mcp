<template>
  <section class="panel tool-panel">
    <div class="panel-header">
      <div class="header-left">
        <h2>Available Tools</h2>
        <span class="tool-count">{{ tools.length }}</span>
      </div>
      <div class="filter-controls">
        <button
          v-for="filter in ['all', 'actions', 'info']"
          :key="filter"
          class="filter-btn"
          :class="{ active: toolFilter === filter }"
          @click="$emit('filter', filter)"
        >
          {{ filter }}
        </button>
      </div>
    </div>
    <div class="panel-content">
      <div class="tool-list" aria-label="Tool list">
        <div class="tool-row" v-for="t in filtered" :key="t.name">
          <div class="row-icon">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M13 2H3a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1V3a1 1 0 00-1-1z" />
              <path d="M8 5v6M5 8h6" />
            </svg>
          </div>
          <div class="row-content">
            <div class="tool-name">
              {{ t.name }}
              <span class="tool-badge">{{ t.type }}</span>
            </div>
            <div class="tool-desc">{{ t.desc }}</div>
          </div>
          <button class="invoke-btn" @click="$emit('invoke', t.name)">
            <span>Run</span>
          </button>
        </div>

        <div v-if="!filtered.length" class="empty-state">
          <span>No tools found matching filter.</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Tool } from '@/composables/useMcpApi'

const props = defineProps<{ tools: Tool[]; toolFilter: string; outlined?: boolean }>()

const filtered = computed(() =>
  props.toolFilter === 'all' ? props.tools : props.tools.filter(t => t.type === props.toolFilter)
)
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  animation: fadeIn 0.35s var(--ease);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 4px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
}

.tool-count {
  font-size: 11px;
  color: var(--muted);
  background: var(--bg-warm);
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid var(--line);
}

/* Filters */
.filter-controls {
  display: flex;
  gap: 4px;
  background: var(--bg-warm);
  padding: 2px;
  border-radius: 6px;
  border: 1px solid var(--line);
}

.filter-btn {
  padding: 4px 10px;
  font-size: 11px;
  color: var(--muted);
  border-radius: 4px;
  text-transform: capitalize;
  transition: all 0.2s ease;
}

.filter-btn:hover {
  color: var(--text);
}

.filter-btn.active {
  background: var(--bg-elevated);
  color: var(--text);
  box-shadow: var(--shadow-xs);
}

/* Tool List */
.panel-content {
  flex: 1;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--bg-warm);
}

.tool-list {
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.tool-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--line-dim);
  transition: background 0.1s ease;
}

.tool-row:last-child {
  border-bottom: none;
}

.tool-row:hover {
  background: var(--panel-light);
}

/* Icon */
.row-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg);
  border: 1px solid var(--line);
  border-radius: 6px;
  color: var(--muted);
}
.row-icon svg { width: 16px; height: 16px; }

/* Content */
.row-content {
  flex: 1;
  min-width: 0; /* Text truncation fix */
}

.tool-name {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text);
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-badge {
  font-family: var(--font-body);
  font-size: 10px;
  color: var(--muted);
  background: var(--bg);
  padding: 1px 5px;
  border-radius: 3px;
  border: 1px solid var(--line-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tool-desc {
  font-size: 12px;
  color: var(--text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Actions */
.invoke-btn {
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  background: var(--bg);
  border: 1px solid var(--line);
  border-radius: 4px;
  transition: all 0.2s ease;
}

.invoke-btn:hover {
  border-color: var(--text-dim);
  background: var(--bg-elevated);
}

.empty-state {
  padding: 40px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
}
</style>
