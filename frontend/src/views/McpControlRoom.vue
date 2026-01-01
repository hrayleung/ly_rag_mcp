<template>
  <div class="app layout-container">
    <!-- Header Spans Full Width -->
    <HeaderBar :latency-label="latencyLabel" />

    <!-- Sidebar (Left) -->
    <StatsSidebar
      :active-panel="activePanel"
      :stats="statsVm"
      :tools-length="tools.value?.data?.length || 0"
      @panel="activePanel = $event"
      @refresh="loadAll"
      @simulate="simulateTrigger"
      @clear-log="clearLog"
    />

    <!-- Main Content Area -->
    <main class="main">
      <div class="top-row">
        <!-- Request Panel (Left Top) -->
         <div class="panel-wrapper requests-wrapper">
          <RequestPanel :requests="requests.value?.data || []" />
        </div>

        <!-- Tool Panel (Right Top) -->
        <div class="panel-wrapper tools-wrapper">
          <ToolPanel
            :tools="tools.value?.data || []"
            :tool-filter="toolFilter"
            @filter="toolFilter = $event"
            @invoke="log(`Invoked ${$event}`)"
          />
        </div>
      </div>

      <!-- Log Panel (Bottom Full Width) -->
      <div class="panel-wrapper logs-wrapper">
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
.app {
  min-height: 100vh;
  display: grid;
  padding: 24px;
  gap: 24px;
  /* Updated Grid Layout */
  grid-template-columns: 240px 1fr;
  grid-template-rows: auto 1fr;
  grid-template-areas:
    "header header"
    "sidebar main";
}

/* Make header span full width but respect grid gap */
/* Wait, header is inside grid area header */

.main {
  grid-area: main;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 0; /* Fix flex overflow */
}

.top-row {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr; /* Requests get more space */
  gap: 24px;
  height: 55%; /* Top section height */
  min-height: 400px;
}

.panel-wrapper {
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.logs-wrapper {
  flex: 1;
  min-height: 300px;
}

/* Responsive */
@media (max-width: 1200px) {
  .app {
    grid-template-columns: 200px 1fr;
  }
}

@media (max-width: 1024px) {
  .app {
    grid-template-columns: 1fr;
    grid-template-areas:
      "header"
      "sidebar"
      "main";
    height: auto;
    overflow-y: auto;
  }

  .main {
    height: auto;
  }

  .top-row {
    grid-template-columns: 1fr;
    height: auto;
  }
}
</style>
