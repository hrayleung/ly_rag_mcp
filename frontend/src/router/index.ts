import { createRouter, createWebHistory } from 'vue-router'
import McpControlRoom from '@/views/McpControlRoom.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: McpControlRoom,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
