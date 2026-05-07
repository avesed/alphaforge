import apiClient from './client'

export interface SettingsMap {
  [key: string]: string
}

export interface SettingValue {
  key: string
  value: string
  updatedAt: string
}

export const settingsApi = {
  getAll: () => apiClient.get<SettingsMap>('/admin/settings'),

  update: (settings: SettingsMap) =>
    apiClient.put<SettingsMap>('/admin/settings', settings),

  get: (key: string) =>
    apiClient.get<SettingValue>(`/admin/settings/${key}`),

  set: (key: string, value: string) =>
    apiClient.put<SettingValue>(`/admin/settings/${key}`, { value }),

  testStockPulse: () =>
    apiClient.get<{ connected: boolean; error?: string }>('/admin/settings/stockpulse/test'),
}
