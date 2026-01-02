<template>
  <div class="app-layout">
    <!-- Top Header -->
    <HeaderBar :latency-label="latencyLabel" class="header-area" />

    <!-- Left Sidebar -->
    <StatsSidebar
      :active-panel="activePanel"
      :stats="statsVm"
      :tools-length="tools.value?.data?.length || 0"
      @panel="activePanel = $event"
      @refresh="loadAll"
      @simulate="simulateTrigger"
      @clear-log="clearLog"
      class="sidebar-area"
    />

    <!-- Main Workspace -->
    <main class="main-area">
      <!-- Upper Grid: Requests & Tools -->
      <div class="upper-grid">
        <div class="panel-container border-right">
          <RequestPanel :requests="requests.value?.data || []" />
        </div>

        <div class="panel-container">
          <ToolPanel
            :tools="tools.value?.data || []"
            :tool-filter="toolFilter"
            @filter="toolFilter = $event"
            @invoke="log(`Invoked ${$event}`)"
          />
        </div>
      </div>

      <!-- Lower Grid: Logs -->
      <div class="lower-grid border-top">
        <LogPanel :logs="logs.value?.data || []" />
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import HeaderBar from '@/components/HeaderBar.vue'
import StatsSidebar from '@/components/StatsSidebar.vue'
import RequestPanel from '@/components/RequestPanel.vue'
import ToolPanel from '@/components/ToolPanel.vue'
import LogPanel from '@/components/LogPanel.vue'
import { useMcpApi } from '@/composables/useMcpApi'
import { usePolling } from '@/composables/usePolling'

const activePanel = ref<'requests' | 'tools' | 'logs'>('requests')
const toolFilter = ref<'all' | 'actions' | 'info'>('all')

const { requests, tools, logs, stats, loadAll } = useMcpApi()
const { start, stop, latencyLabel } = usePolling(3000, loadAll)

const statsVm = computed(() => ({
  uptime: formatUptime(stats.value?.data?.uptime_seconds),
  rpm: stats.value?.data?.rpm ?? null,
  errors: stats.value?.data?.errors ?? null,
}))

function formatUptime(seconds?: number) {
  if (seconds === undefined || seconds === null) return '—'
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return [d ? `${d}d` : null, h ? `${h}h` : null, m ? `${m}m` : null, `${s}s`].filter(Boolean).join(' ')
}

function clearLog() {
  if (logs.value?.data) logs.value.data = []
}
function simulateTrigger() { log('Triggered sample tool') }
function log(msg: string) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`
  const arr = logs.value?.data || []
  logs.value = { ...logs.value, data: [line, ...arr].slice(0, 200) }
}

onMounted(() => {
  loadAll()
  start()
})

onBeforeUnmount(() => stop())
</script>

<style scoped>
/* Main Grid Construction */
.app-layout {
  display: grid;
  height: 100vh;
  width: 100vw;
  grid-template-columns: 240px 1fr;
  grid-template-rows: 48px 1fr;
  grid-template-areas:
    "header header"
    "sidebar main";
  background: var(--bg);
  overflow: hidden;
}

/* Header */
.header-area {
  grid-area: header;
  border-bottom: 1px solid var(--border);
  z-index: 10;
  background: var(--bg); /* Ensure opaque */
}

/* Sidebar */
.sidebar-area {
  grid-area: sidebar;
  border-right: 1px solid var(--border);
  background: var(--bg-panel);
  overflow-y: auto;
}

/* Main Area */
.main-area {
  grid-area: main;
  display: grid;
  grid-template-rows: 60% 40%;
  min-height: 0;
  background: var(--bg);
}

/* Upper Grid Split (Requests 65% / Tools 35%) */
.upper-grid {
  display: grid;
  grid-template-columns: 65% 35%;
  min-height: 0;
}

.panel-container {
  height: 100%;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column;
}

/* Lower Grid (Logs) */
.lower-grid {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* Borders Utilities */
.border-right {
  border-right: 1px solid var(--border);
}
.border-top {
  border-top: 1px solid var(--border);
}

/* Responsive */
@media (max-width: 1200px) {
  .upper-grid {
    grid-template-columns: 55% 45%;
  }
}

@media (max-width: 1024px) {
  .app-layout {
    grid-template-columns: 1fr;
    grid-template-rows: 48px auto 1fr;
    grid-template-areas:
      "header"
      "sidebar"
      "main";
    overflow-y: auto; /* Allow full page scroll */
  }

  .sidebar-area {
    height: auto;
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .main-area {
    display: flex;
    flex-direction: column;
    height: auto;
  }

  .upper-grid {
    display: flex;
    flex-direction: column;
    height: auto;
  }

  .panel-container {
    height: 500px;
  }

  .border-right {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .lower-grid {
    height: 400px;
  }
}
</style>
