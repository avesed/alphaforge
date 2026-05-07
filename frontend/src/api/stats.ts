import apiClient from './client'
import type { DashboardStats } from '@/types'

export const statsApi = {
  get: () => apiClient.get<DashboardStats>('/admin/stats'),
}
