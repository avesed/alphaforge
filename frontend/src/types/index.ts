export interface User {
  id: number
  email: string
  displayName: string | null
  role: 'admin' | 'user'
  locale: string
}

export interface TokenResponse {
  accessToken: string
  refreshToken: string
  tokenType: string
}

export interface ApiError {
  detail: string
}

export type Market = 'us' | 'cn' | 'hk'

export interface PredictionModel {
  id: string
  market: Market
  modelDate: string
  modelType: string
  forwardDays: number
  ic: number | null
  icir: number | null
  ndcg: number | null
  qualityPassed: boolean
  featureCount: number | null
  symbolCount: number | null
  createdAt: string
}

export interface StockPrediction {
  id: number
  market: Market
  predictionDate: string
  symbol: string
  predictedScore: number
  percentileRank: number
  predictedDirection: 'up' | 'down' | 'neutral'
  actualReturn: number | null
  forwardDays: number
}

export interface PredictionUniverse {
  id: number
  name: string
  market: Market
  universeType: string
  symbols: string[]
  createdAt: string
}

export interface BacktestResult {
  id: number
  strategyType: string
  market: Market
  status: 'pending' | 'running' | 'completed' | 'failed'
  results: Record<string, unknown> | null
  createdAt: string
}

export interface DiscoveredFactor {
  id: number
  name: string
  expression: string
  market: Market
  ic: number
  icir: number
  isActive: boolean
  createdAt: string
}

// Task tracking
export interface TaskStatus {
  taskId: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  market?: string
  progress?: number
  result?: Record<string, unknown>
  error?: string
  startedAt?: string
  completedAt?: string
}

// API Consumer
export interface ApiConsumer {
  id: string
  name: string
  apiKeyPrefix: string
  description?: string
  isActive: boolean
  rateLimit: number
  lastUsedAt?: string
  createdAt: string
}

export interface ApiConsumerWithKey extends ApiConsumer {
  rawApiKey: string
}

// Scheduler
export interface SchedulerJob {
  id: string
  name: string
  nextRunTime: string | null
  trigger: string | null
}

export interface SchedulerStatus {
  isLeader: boolean
  running: boolean
  jobCount: number
  jobs: SchedulerJob[]
}

// Stats
export interface DashboardStats {
  userCount: number
  consumerCount: number
  schedulerIsLeader: boolean
  schedulerJobCount: number
  stockpulseConnected: boolean
  redisConnected: boolean
}

// Performance
export interface PerformanceMetric {
  date: string
  ic: number
  hitRate: number
  topReturn: number
  bottomReturn: number
  spread: number
  symbolCount: number
}

// RD-Agent
export interface RDAgentStatus {
  market: string
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'error'
  currentRound?: number
  maxRounds?: number
  factorsFound?: number
  error?: string
}
