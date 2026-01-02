import { ref } from 'vue'

export type Tool = { name: string; type: string; desc: string }
export type RequestEntry = { status: string; route: string; latency: number; tool: string; user: string; ts?: number; code?: number }
export type StatsPayload = { uptime_seconds?: number; rpm?: number; errors?: number; index?: Record<string, unknown> }

type ApiState<T> = {
  data: T | null
  loading: boolean
  error: string | null
}

const API_BASE = import.meta.env.VITE_API_BASE || '/api/mcp'

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function useMcpApi() {
  const requests = ref<ApiState<RequestEntry[]>>({ data: null, loading: false, error: null })
  const tools = ref<ApiState<Tool[]>>({ data: null, loading: false, error: null })
  const logs = ref<ApiState<string[]>>({ data: null, loading: false, error: null })
  const stats = ref<ApiState<StatsPayload>>({ data: null, loading: false, error: null })

  async function loadAll() {
    requests.value.loading = tools.value.loading = logs.value.loading = stats.value.loading = true
    requests.value.error = tools.value.error = logs.value.error = stats.value.error = null
    try {
      const [reqData, toolData, logData, statsData] = await Promise.all([
        fetchJSON<RequestEntry[]>('/requests'),
        fetchJSON<Tool[]>('/tools'),
        fetchJSON<string[]>('/logs'),
        fetchJSON<StatsPayload>('/stats'),
      ])
      requests.value.data = reqData || []
      tools.value.data = toolData || []
      logs.value.data = Array.isArray(logData) ? logData.slice(-200).reverse() : []
      stats.value.data = statsData || {}
    } catch (e: any) {
      const msg = e?.message || 'Failed to load MCP data'
      requests.value.error = tools.value.error = logs.value.error = stats.value.error = msg
    } finally {
      requests.value.loading = tools.value.loading = logs.value.loading = stats.value.loading = false
    }
  }

  return { requests, tools, logs, stats, loadAll }
}
