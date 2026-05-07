import apiClient from './client'
import type {
  TaskStatus,
  PredictionModel,
  StockPrediction,
  PerformanceMetric,
} from '@/types'

export interface TriggerPredictionParams {
  forceRetrain?: boolean
  forwardDays?: number
}

export interface PredictionLatestResponse {
  market: string
  predictionDate: string
  modelDate: string
  forwardDays: number
  predictions: StockPrediction[]
  qualityPassed: boolean
}

export interface PredictionHistoryResponse {
  market: string
  dates: string[]
  predictions: StockPrediction[]
}

export interface FeatureImportance {
  feature: string
  importance: number
  rank: number
}

export interface AccuracyResponse {
  market: string
  days: number
  overall: {
    hitRate: number
    avgIc: number
    avgIcir: number
  }
  daily: Array<{
    date: string
    hitRate: number
    ic: number
    symbolCount: number
  }>
}

export interface IcDecayResponse {
  market: string
  days: number[]
  icValues: number[]
}

export interface TurnoverResponse {
  market: string
  dates: string[]
  turnoverRates: number[]
  avgTurnover: number
}

export interface AttributionResponse {
  market: string
  categories: Array<{
    name: string
    contribution: number
    featureCount: number
  }>
  topFeatures: FeatureImportance[]
}

export interface PredictionDatesResponse {
  market: string
  dates: string[]
}

export const predictionsApi = {
  // Trigger
  triggerPrediction: (market: string, forceRetrain = false, forwardDays = 5) =>
    apiClient.post<TaskStatus>(`/predictions/${market}/run`, {
      force_retrain: forceRetrain,
      forward_days: forwardDays,
    }),

  getTaskStatus: (taskId: string) =>
    apiClient.get<TaskStatus>(`/predictions/tasks/${taskId}`),

  // Results
  getLatest: (market: string, topN = 50) =>
    apiClient.get<PredictionLatestResponse>(`/predictions/${market}/latest`, {
      params: { top_n: topN },
    }),

  getHistory: (market: string, days = 30) =>
    apiClient.get<PredictionHistoryResponse>(`/predictions/${market}/history`, {
      params: { days },
    }),

  // Models
  getModels: (market?: string) =>
    apiClient.get<PredictionModel[]>('/predictions/models', {
      params: market ? { market } : {},
    }),

  getFeatureImportance: (modelId: string) =>
    apiClient.get<FeatureImportance[]>(
      `/predictions/models/${modelId}/feature-importance`
    ),

  updateModelQuality: (modelId: string, passed: boolean) =>
    apiClient.put(`/predictions/models/${modelId}/quality`, {
      quality_passed: passed,
    }),

  // Analytics
  getAccuracy: (market: string, days = 30) =>
    apiClient.get<AccuracyResponse>(`/predictions/${market}/accuracy`, {
      params: { days },
    }),

  getPerformance: (market: string, days = 30) =>
    apiClient.get<PerformanceMetric[]>(`/predictions/${market}/performance`, {
      params: { days },
    }),

  getIcDecay: (market: string, days = 30) =>
    apiClient.get<IcDecayResponse>(`/predictions/${market}/ic-decay`, {
      params: { days },
    }),

  getTurnover: (market: string, days = 30, topN = 20) =>
    apiClient.get<TurnoverResponse>(`/predictions/${market}/turnover`, {
      params: { days, top_n: topN },
    }),

  getAttribution: (market: string, days = 30, topN = 20) =>
    apiClient.get<AttributionResponse>(`/predictions/${market}/attribution`, {
      params: { days, top_n: topN },
    }),

  getPredictionDates: (market: string, nDates = 10) =>
    apiClient.get<PredictionDatesResponse>(
      `/predictions/${market}/prediction-dates`,
      { params: { n_dates: nDates } }
    ),

  // Backfill
  backfillReturns: () =>
    apiClient.post<TaskStatus>('/predictions/backfill-returns'),
}
