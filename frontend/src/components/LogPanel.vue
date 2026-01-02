<template>
  <section class="panel log-panel">
    <div class="panel-header">
      <h2 class="section-label">SYSTEM LOGS</h2>
      <div class="actions">
        <!-- Optional Actions if needed -->
      </div>
    </div>
    <div class="panel-content">
      <div class="log-container" ref="logContainer">
        <div v-for="(line, i) in logs" :key="i" class="log-line">
          <span class="gutter">{{ i + 1 }}</span>
          <span class="timestamp">{{ getTimestamp(line) }}</span>
          <span class="divider">|</span>
          <span class="message" :class="getMessageClass(line)">{{ getMessage(line) }}</span>
        </div>
        <div v-if="!logs.length" class="empty-state">
           <span class="cursor">_</span> Waiting for server events...
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ logs: string[] }>()
const logContainer = ref<HTMLElement | null>(null)

function getTimestamp(line: string): string {
  const match = line.match(/\[([^\]]+)\]/)
  return match ? match[1] : '' // Just return the time string
}

function getMessage(line: string): string {
  return line.replace(/^\[[^\]]+\]\s*/, '')
}

function getMessageClass(line: string): string {
  if (line.toLowerCase().includes('error')) return 'msg-error'
  if (line.toLowerCase().includes('warn')) return 'msg-warn'
  if (line.includes('Invoked')) return 'msg-info'
  return ''
}
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

.section-label {
  margin: 0;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--fg-muted);
}

.panel-content {
  flex: 1;
  overflow: hidden;
  background: #000; /* Pure black terminal */
}

.log-container {
  height: 100%;
  overflow-y: auto;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.5;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
}

.log-line {
  display: flex;
  gap: 12px;
  color: var(--fg-muted);
}

.gutter {
  color: #333;
  user-select: none;
  width: 24px;
  text-align: right;
  flex-shrink: 0;
}

.timestamp {
  color: var(--fg-dim);
  flex-shrink: 0;
  min-width: 60px;
  user-select: none;
}

.divider {
  color: #333;
  user-select: none;
}

.message {
  word-break: break-word;
  color: var(--fg);
}

/* Semantics */
.msg-error { color: var(--error); }
.msg-warn { color: var(--warning); }
.msg-info { color: var(--brand); }

.empty-state {
  color: var(--fg-dim);
  padding: 20px 0;
  font-family: var(--font-mono);
}

.cursor {
  animation: blink 1s step-end infinite;
  color: var(--fg);
}

@keyframes blink {
  50% { opacity: 0; }
}
</style>
