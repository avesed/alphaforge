import apiClient from './client'

export interface FactorData {
  symbol: string
  market: string
  date: string
  factors: Record<string, number | null>
}

export interface FactorSummary {
  symbol: string
  market: string
  factorCount: number
  dateRange: {
    start: string
    end: string
  }
  statistics: Record<
    string,
    {
      mean: number
      std: number
      min: number
      max: number
    }
  >
}

export interface IcResult {
  factor: string
  ic: number
  icir: number
  pValue: number
}

export interface CsRankResult {
  date: string
  rankings: Array<{
    symbol: string
    rank: number
    value: number
  }>
}

export interface ExpressionResult {
  expression: string
  values: Array<{
    date: string
    value: number | null
  }>
}

export interface ValidationResult {
  valid: boolean
  expression: string
  error?: string
}

export const qlibApi = {
  getFactors: (symbol: string, market?: string) =>
    apiClient.get<FactorData[]>(`/factors/${symbol}`, {
      params: market ? { market } : {},
    }),

  getFactorSummary: (symbol: string, market?: string) =>
    apiClient.get<FactorSummary>(`/factors/${symbol}/summary`, {
      params: market ? { market } : {},
    }),

  computeIc: (params: Record<string, unknown>) =>
    apiClient.post<IcResult[]>('/factors/ic', params),

  csRank: (params: Record<string, unknown>) =>
    apiClient.post<CsRankResult>('/factors/cs-rank', params),

  evaluateExpression: (params: Record<string, unknown>) =>
    apiClient.post<ExpressionResult>('/expression/evaluate', params),

  batchExpression: (params: Record<string, unknown>) =>
    apiClient.post<ExpressionResult[]>('/expression/batch', params),

  validateExpression: (expression: string) =>
    apiClient.post<ValidationResult>('/expression/validate', { expression }),
}
