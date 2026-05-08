import apiClient from './client'
import type {
  TaskStatus,
  PredictionModel,
  StockPrediction,
} from '@/types'

export interface TriggerPredictionParams {
  forceRetrain?: boolean
  forwardDays?: number
}

export interface PredictionLatestResponse {
  market: string
  predictionDate: string | null
  count: number
  predictions: StockPrediction[]
}

export interface PredictionHistoryResponse {
  market: string
  days: number
  count: number
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
  totalPredictions: number
  directionAccuracy: number | null
  ic: number | null
  icir: number | null
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformModel(m: any): PredictionModel {
  return {
    id: m.id,
    market: m.market,
    modelDate: m.model_date ?? m.modelDate,
    modelType: m.model_type ?? m.modelType ?? 'ranking',
    forwardDays: m.forward_days ?? m.forwardDays ?? 5,
    ic: m.ic ?? null,
    icir: m.icir ?? null,
    ndcg: m.ndcg ?? null,
    qualityPassed: m.quality === 'approved' || m.quality_passed === true || m.qualityPassed === true,
    featureCount: m.feature_count ?? m.featureCount ?? null,
    symbolCount: m.symbol_count ?? m.symbolCount ?? null,
    createdAt: m.created_at ?? m.createdAt ?? '',
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformPrediction(p: any): StockPrediction {
  const upProb = p.up_probability ?? p.upProbability ?? 0.5
  let direction: 'up' | 'down' | 'neutral' = 'neutral'
  if (upProb > 0.55) direction = 'up'
  else if (upProb < 0.45) direction = 'down'

  return {
    id: p.id,
    market: p.market,
    predictionDate: p.prediction_date ?? p.predictionDate ?? '',
    symbol: p.symbol,
    predictedScore: p.rank_score ?? p.predictedScore ?? 0,
    percentileRank: (p.percentile_rank ?? p.percentileRank ?? 0) * 100,
    predictedDirection: p.predicted_direction ?? p.predictedDirection ?? direction,
    actualReturn: p.actual_return ?? p.actualReturn ?? null,
    forwardDays: p.forward_days ?? p.forwardDays ?? 5,
  }
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
    apiClient.get(`/predictions/${market}/latest`, {
      params: { top_n: topN },
    }).then(r => {
      const raw = r.data
      const predictions = (raw.predictions ?? []).map(transformPrediction)
      const data: PredictionLatestResponse = {
        market: raw.market,
        predictionDate: predictions.length > 0 ? predictions[0].predictionDate : null,
        count: raw.count ?? predictions.length,
        predictions,
      }
      return { ...r, data }
    }),

  getHistory: (market: string, days = 30) =>
    apiClient.get(`/predictions/${market}/history`, {
      params: { days },
    }).then(r => {
      const raw = r.data
      const data: PredictionHistoryResponse = {
        market: raw.market,
        days: raw.days,
        count: raw.count,
        predictions: (raw.predictions ?? []).map(transformPrediction),
      }
      return { ...r, data }
    }),

  // Models
  getModels: (market?: string) =>
    apiClient.get('/predictions/models', {
      params: market ? { market } : {},
    }).then(r => ({
      ...r,
      data: (r.data.models ?? r.data ?? []).map(transformModel) as PredictionModel[],
    })),

  getFeatureImportance: (modelId: string) =>
    apiClient.get(`/predictions/models/${modelId}/feature-importance`).then(r => {
      const raw = r.data
      const top30 = raw.top30 ?? {}
      let features: FeatureImportance[]
      if (Array.isArray(top30)) {
        features = top30
      } else {
        features = Object.entries(top30).map(([feature, importance], idx) => ({
          feature,
          importance: importance as number,
          rank: idx + 1,
        }))
      }
      return { ...r, data: features }
    }),

  updateModelQuality: (modelId: string, passed: boolean) =>
    apiClient.put(`/predictions/models/${modelId}/quality`, {
      quality_passed: passed,
    }),

  // Analytics
  getAccuracy: (market: string, days = 30) =>
    apiClient.get(`/predictions/${market}/accuracy`, {
      params: { days },
    }).then(r => {
      const raw = r.data
      const data: AccuracyResponse = {
        market: raw.market,
        days: raw.days,
        totalPredictions: raw.total_predictions ?? raw.totalPredictions ?? 0,
        directionAccuracy: raw.direction_accuracy ?? raw.directionAccuracy ?? null,
        ic: raw.ic ?? null,
        icir: raw.icir ?? null,
      }
      return { ...r, data }
    }),

  getPerformance: (market: string, days = 30) =>
    apiClient.get(`/predictions/${market}/performance`, {
      params: { days },
    }).then(r => ({
      ...r,
      data: r.data.data ?? r.data ?? [],
    })),

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
