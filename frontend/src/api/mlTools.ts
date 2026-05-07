import apiClient from './client'
import type { TaskStatus } from '@/types'

export interface ProfileConfig {
  market: string
  forwardDays?: number
  [key: string]: unknown
}

export interface TrainConfig {
  market: string
  forwardDays?: number
  forceRetrain?: boolean
  [key: string]: unknown
}

export interface ValidateConfig {
  taskId: string
  folds?: number
  [key: string]: unknown
}

export interface RollingBacktestConfig {
  market: string
  startDate?: string
  endDate?: string
  retrainInterval?: number
  [key: string]: unknown
}

export interface DeployResult {
  taskId: string
  modelId: string
  status: string
}

export const mlToolsApi = {
  profile: (market: string, config?: Record<string, unknown>) =>
    apiClient.post<TaskStatus>('/ml-tools/profile', { market, ...config }),

  train: (market: string, config?: Record<string, unknown>) =>
    apiClient.post<TaskStatus>('/ml-tools/train', { market, ...config }),

  getTaskStatus: (taskId: string) =>
    apiClient.get<TaskStatus>(`/ml-tools/tasks/${taskId}`),

  validate: (taskId: string, config?: Record<string, unknown>) =>
    apiClient.post<TaskStatus>('/ml-tools/validate', {
      task_id: taskId,
      ...config,
    }),

  rollingBacktest: (market: string, config: Record<string, unknown>) =>
    apiClient.post<TaskStatus>('/ml-tools/rolling-backtest', {
      market,
      ...config,
    }),

  deploy: (taskId: string) =>
    apiClient.post<DeployResult>('/ml-tools/deploy', { task_id: taskId }),
}
