import apiClient from './client'
import type { RDAgentStatus, DiscoveredFactor } from '@/types'

export interface StartRDAgentParams {
  maxRounds?: number
  universeId?: string
}

export interface RDAgentFactorsResponse {
  factors: DiscoveredFactor[]
  total: number
}

export const rdagentApi = {
  start: (market: string, maxRounds = 30, universeId?: string) =>
    apiClient.post<RDAgentStatus>(`/rdagent/${market}/start`, {
      max_rounds: maxRounds,
      universe_id: universeId,
    }),

  getStatus: (market: string) =>
    apiClient.get<RDAgentStatus>(`/rdagent/${market}/status`),

  stop: (market: string) =>
    apiClient.post<RDAgentStatus>(`/rdagent/${market}/stop`),

  getFactors: (market?: string) =>
    apiClient.get<RDAgentFactorsResponse>('/rdagent/factors', {
      params: market ? { market } : {},
    }),

  toggleFactor: (factorId: string, isActive: boolean) =>
    apiClient.put<DiscoveredFactor>(`/rdagent/factors/${factorId}`, {
      is_active: isActive,
    }),
}
