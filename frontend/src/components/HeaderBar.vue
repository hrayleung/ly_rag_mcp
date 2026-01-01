<template>
  <header class="header">
    <div class="header-content">
      <div class="brand">
        <div class="logo-mark">
          <svg viewBox="0 0 24 24" fill="none" class="brand-icon">
            <path stroke="currentColor" stroke-width="1.5" d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <div>
          <h1>Ly</h1>
        </div>
      </div>
      <div class="header-right">
        <div class="status-indicator" :class="{ ok: true }">
          <span class="status-dot"></span>
          <span class="status-label">Live</span>
        </div>
        <div class="latency-badge">
          <span class="latency-label">Latency</span>
          <span class="latency-value">{{ latencyLabel }}</span>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
const props = defineProps<{ latencyLabel: string }>()
</script>

<style scoped>
.header {
  grid-area: header;
  display: flex;
  align-items: center;
  background: transparent;
  width: 100%;
  animation: fadeIn var(--duration) var(--ease);
}

.header-content {
  width: 100%;
  padding: 0 4px; /* Align with sidebar padding */
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Minimalist Logo Mark */
.logo-mark {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text);
  opacity: 0.9;
}

.brand-icon {
  width: 20px;
  height: 20px;
}

h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text);
}

/* Right Section */
.header-right {
  display: flex;
  align-items: center;
  gap: 24px;
}

/* Status Indicator - Minimal dot */
.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 6px;
  height: 6px;
  background: var(--success);
  border-radius: 50%;
  position: relative;
}

.status-dot::after {
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 50%;
  border: 1px solid var(--success);
  opacity: 0.3;
}

.status-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-dim);
}

/* Latency - Plain text */
.latency-badge {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.latency-label {
  font-size: 12px;
  color: var(--muted);
}

.latency-value {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-dim);
}
</style>
