<template>
  <section class="panel log-panel">
    <div class="panel-header">
      <h2>Server Logs</h2>
      <div class="actions">
        <button v-if="logs.length" class="text-btn" @click="$emit('clear-log')">Clear</button>
      </div>
    </div>
    <div class="panel-content">
      <div class="log-container" ref="logContainer">
        <div v-for="(line, i) in logs" :key="i" class="log-line">
          <span class="timestamp">{{ getTimestamp(line) }}</span>
          <span class="message">{{ getMessage(line) }}</span>
        </div>
        <div v-if="!logs.length" class="empty-state">
           Waiting for server events...
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ logs: string[]; outlined?: boolean }>()
const logContainer = ref<HTMLElement | null>(null)

function getTimestamp(line: string): string {
  const match = line.match(/\[([^\]]+)\]/)
  return match ? match[1] : '' // Just return the time string
}

function getMessage(line: string): string {
  return line.replace(/^\[[^\]]+\]\s*/, '')
}
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  animation: fadeIn 0.4s var(--ease);
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

.text-btn {
  font-size: 11px;
  color: var(--muted);
  text-decoration: underline;
  text-decoration-color: transparent;
  transition: all 0.2s ease;
}

.text-btn:hover {
  color: var(--text);
  text-decoration-color: var(--line);
}

.panel-content {
  flex: 1;
  overflow: hidden;
  background: black; /* Terminal feel */
  border: 1px solid var(--line);
  border-radius: var(--radius);
}

.log-container {
  height: 100%;
  overflow-y: auto;
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.6;
  padding: 12px;
  display: flex;
  flex-direction: column;
}

.log-line {
  display: flex;
  gap: 12px;
  color: var(--text-dim);
}

.timestamp {
  color: var(--muted);
  flex-shrink: 0;
  min-width: 60px;
  user-select: none;
}

.message {
  word-break: break-word;
  color: #D4D4D8; /* Zinc-300 */
}

.empty-state {
  color: var(--muted-dim);
  font-style: italic;
  padding: 20px 0;
  text-align: center;
}
</style>
