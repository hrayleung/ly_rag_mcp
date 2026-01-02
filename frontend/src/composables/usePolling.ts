import { onBeforeUnmount, ref } from 'vue'

export function usePolling(intervalMs: number, task: () => Promise<void>) {
  const latencyLabel = ref('—')
  let timer: ReturnType<typeof setInterval> | null = null

  async function run() {
    const start = performance.now()
    await task()
    latencyLabel.value = `${Math.round(performance.now() - start)} ms`
  }

  function start() {
    run()
    timer = setInterval(run, intervalMs)
  }

  function stop() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  onBeforeUnmount(stop)

  return { start, stop, latencyLabel }
}
