import apiClient from './client'
import type { ApiConsumer, ApiConsumerWithKey } from '@/types'

export interface CreateConsumerParams {
  name: string
  description?: string
}

export interface UpdateConsumerParams {
  name?: string
  description?: string
  rateLimit?: number
  isActive?: boolean
}

export const consumersApi = {
  list: () => apiClient.get<ApiConsumer[]>('/admin/consumers'),

  create: (name: string, description?: string) =>
    apiClient.post<ApiConsumerWithKey>('/admin/consumers', {
      name,
      description,
    }),

  update: (id: string, data: UpdateConsumerParams) =>
    apiClient.put<ApiConsumer>(`/admin/consumers/${id}`, data),

  deactivate: (id: string) =>
    apiClient.put<ApiConsumer>(`/admin/consumers/${id}`, { isActive: false }),

  delete: (id: string) =>
    apiClient.delete(`/admin/consumers/${id}`),
}
