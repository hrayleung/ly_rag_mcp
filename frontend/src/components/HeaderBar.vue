<template>
  <header class="header">
    <div class="header-content">
      <div class="brand">
        <div class="logo-mark">
          <svg viewBox="0 0 24 24" fill="none" class="brand-icon">
            <path stroke="currentColor" stroke-width="1.5" d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <div class="brand-text">
          <h1>Ly</h1>
          <span class="brand-divider">/</span>
          <span class="brand-context">Factory</span>
        </div>
      </div>
      <div class="header-right">
        <div class="status-indicator">
          <span class="status-dot"></span>
          <span class="status-label mono">LIVE</span>
        </div>
        <div class="latency-badge">
          <span class="latency-label">LATENCY</span>
          <span class="latency-value mono">{{ latencyLabel }}</span>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{ latencyLabel: string }>()
</script>

<style scoped>
.header {
  height: 100%;
  display: flex;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
}

.header-content {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* Brand Section */
.brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.logo-mark {
  color: var(--fg);
  display: flex;
  align-items: center;
}

.brand-icon {
  width: 20px;
  height: 20px;
}

.brand-text {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

h1 {
  margin: 0;
  font-weight: 600;
  color: var(--fg);
  font-size: 14px;
}

.brand-divider {
  color: var(--border-active);
}

.brand-context {
  color: var(--fg-muted);
}

/* Right Section */
.header-right {
  display: flex;
  align-items: center;
  gap: 32px;
}

/* Indicators */
.status-indicator, .latency-badge {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 6px;
  height: 6px;
  background: var(--success);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--success); /* Small glow */
}

.status-label, .latency-label {
  font-size: 11px;
  color: var(--fg-muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.latency-value {
  font-size: 12px;
  color: var(--fg);
}
</style>
