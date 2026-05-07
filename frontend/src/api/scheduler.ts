import apiClient from './client'
import type { SchedulerStatus } from '@/types'

export const schedulerApi = {
  getJobs: () => apiClient.get<SchedulerStatus>('/admin/scheduler/jobs'),

  triggerJob: (jobId: string) =>
    apiClient.post(`/admin/scheduler/trigger/${jobId}`),

  relinquish: () => apiClient.post('/admin/scheduler/relinquish'),
}
