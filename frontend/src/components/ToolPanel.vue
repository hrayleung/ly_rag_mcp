<template>
  <section class="panel tool-panel">
    <div class="panel-header">
      <div class="header-left">
        <h2 class="section-label">AVAILABLE TOOLS</h2>
        <span class="tool-count mono">{{ tools.length }}</span>
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

    <div class="list-container">
      <div class="tool-list" aria-label="Tool list">
        <div class="tool-row" v-for="t in filtered" :key="t.name">
          <div class="row-content">
            <div class="tool-name">
              <span class="name-text">{{ t.name }}</span>
              <span class="badge mono">{{ t.type }}</span>
            </div>
            <div class="tool-desc">{{ t.desc }}</div>
          </div>
          <button class="invoke-btn" @click="$emit('invoke', t.name)">
            RUN
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

const props = defineProps<{ tools: Tool[]; toolFilter: string }>()

const filtered = computed(() =>
  props.toolFilter === 'all' ? props.tools : props.tools.filter(t => t.type === props.toolFilter)
)
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

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-label {
  margin: 0;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--fg-muted);
}

.tool-count {
  font-size: 10px;
  color: var(--fg-muted);
  background: var(--bg-hover);
  padding: 1px 4px;
  border-radius: 2px;
}

/* Filters */
.filter-controls {
  display: flex;
  gap: 16px;
}

.filter-btn {
  font-size: 10px;
  color: var(--fg-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
  padding: 4px 0;
  transition: all 0.2s ease;
}

.filter-btn:hover {
  color: var(--fg);
}

.filter-btn.active {
  color: var(--fg);
  border-bottom: 1px solid var(--fg); /* Minimal underscore */
}

/* List */
.list-container {
  flex: 1;
  overflow-y: auto;
  background: var(--bg-panel);
}

.tool-list {
  display: flex;
  flex-direction: column;
}

.tool-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  gap: 16px;
}

.tool-row:hover {
  background: var(--bg-hover);
}

.row-content {
  flex: 1;
  min-width: 0;
}

.tool-name {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.name-text {
  font-size: 13px;
  font-weight: 500;
  color: var(--fg);
}

.badge {
  font-size: 9px;
  color: var(--fg-muted);
  border: 1px solid var(--border-active);
  padding: 1px 4px;
  border-radius: 2px;
  text-transform: uppercase;
}

.tool-desc {
  font-size: 12px;
  color: var(--fg-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Actions */
.invoke-btn {
  font-size: 10px;
  font-weight: 600;
  color: var(--fg);
  border: 1px solid var(--border-active);
  padding: 4px 8px;
  border-radius: 2px;
  background: transparent;
  transition: all 0.1s ease;
}

.invoke-btn:hover {
  background: var(--fg);
  color: var(--bg);
  border-color: var(--fg);
}

.empty-state {
  padding: 40px;
  text-align: center;
  color: var(--fg-muted);
  font-size: 13px;
}
</style>
