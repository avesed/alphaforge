import apiClient from './client'
import type { TaskStatus, BacktestResult } from '@/types'

export interface BacktestConfig {
  strategyType?: string
  startDate?: string
  endDate?: string
  topK?: number
  benchmark?: string
  [key: string]: unknown
}

export interface AgentBacktestParams {
  startDate?: string
  endDate?: string
  maxRounds?: number
  [key: string]: unknown
}

export interface BacktestListResponse {
  backtests: BacktestResult[]
  total: number
}

export const backtestsApi = {
  startBacktest: (market: string, config: BacktestConfig) =>
    apiClient.post<TaskStatus>(`/predictions/${market}/backtest`, config),

  startAgentBacktest: (market: string, params: AgentBacktestParams) =>
    apiClient.post<TaskStatus>(
      `/predictions/${market}/agent-backtest`,
      params
    ),

  getTaskStatus: (taskId: string) =>
    apiClient.get<TaskStatus>(`/predictions/backtests/tasks/${taskId}`),

  list: (market: string, limit = 50) =>
    apiClient.get<BacktestListResponse>(`/predictions/${market}/backtests`, {
      params: { limit },
    }),

  getDetail: (backtestId: string) =>
    apiClient.get<BacktestResult>(`/predictions/backtests/${backtestId}`),

  delete: (backtestId: string) =>
    apiClient.delete(`/predictions/backtests/${backtestId}`),
}
