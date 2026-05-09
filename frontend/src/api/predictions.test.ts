import { transformModel, transformPrediction } from './predictions'

// ---------------------------------------------------------------------------
// transformModel()
// ---------------------------------------------------------------------------
describe('transformModel()', () => {
  it('maps snake_case fields to camelCase', () => {
    const result = transformModel({
      id: 'model-1',
      market: 'us',
      model_date: '2024-03-15',
      model_type: 'ranking',
      forward_days: 5,
      ic: 0.05,
      icir: 0.3,
      ndcg: 0.8,
      quality: 'approved',
      feature_count: 120,
      symbol_count: 500,
      created_at: '2024-03-15T10:00:00Z',
    })

    expect(result.modelDate).toBe('2024-03-15')
    expect(result.modelType).toBe('ranking')
    expect(result.forwardDays).toBe(5)
    expect(result.ic).toBe(0.05)
    expect(result.icir).toBe(0.3)
    expect(result.ndcg).toBe(0.8)
    expect(result.qualityPassed).toBe(true)
    expect(result.featureCount).toBe(120)
    expect(result.symbolCount).toBe(500)
    expect(result.createdAt).toBe('2024-03-15T10:00:00Z')
  })

  it('accepts camelCase fields as well', () => {
    const result = transformModel({
      id: 'model-2',
      market: 'cn',
      modelDate: '2024-01-01',
      modelType: 'direction',
      forwardDays: 10,
      ic: null,
      icir: null,
      ndcg: null,
      qualityPassed: true,
      featureCount: 80,
      symbolCount: 300,
      createdAt: '2024-01-01T00:00:00Z',
    })

    expect(result.modelDate).toBe('2024-01-01')
    expect(result.modelType).toBe('direction')
    expect(result.forwardDays).toBe(10)
    expect(result.qualityPassed).toBe(true)
  })

  it('uses defaults when fields are missing', () => {
    const result = transformModel({
      id: 'model-3',
      market: 'hk',
    })

    expect(result.modelType).toBe('ranking')
    expect(result.forwardDays).toBe(5)
    expect(result.ic).toBeNull()
    expect(result.icir).toBeNull()
    expect(result.ndcg).toBeNull()
    expect(result.featureCount).toBeNull()
    expect(result.symbolCount).toBeNull()
    expect(result.createdAt).toBe('')
  })

  it('sets qualityPassed=true when quality is "approved"', () => {
    const result = transformModel({ id: '1', market: 'us', quality: 'approved' })
    expect(result.qualityPassed).toBe(true)
  })

  it('sets qualityPassed=true when quality_passed is true', () => {
    const result = transformModel({ id: '1', market: 'us', quality_passed: true })
    expect(result.qualityPassed).toBe(true)
  })

  it('sets qualityPassed=false when quality is not approved and flags are absent', () => {
    const result = transformModel({ id: '1', market: 'us', quality: 'rejected' })
    expect(result.qualityPassed).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// transformPrediction()
// ---------------------------------------------------------------------------
describe('transformPrediction()', () => {
  it('maps snake_case fields to camelCase', () => {
    const result = transformPrediction({
      id: 1,
      market: 'us',
      prediction_date: '2024-03-15',
      symbol: 'AAPL',
      rank_score: 0.85,
      percentile_rank: 0.95,
      up_probability: 0.7,
      actual_return: 0.02,
      forward_days: 5,
    })

    expect(result.predictionDate).toBe('2024-03-15')
    expect(result.symbol).toBe('AAPL')
    expect(result.predictedScore).toBe(0.85)
    expect(result.percentileRank).toBe(95)
    expect(result.actualReturn).toBe(0.02)
    expect(result.forwardDays).toBe(5)
  })

  it('multiplies percentile_rank by 100', () => {
    const result = transformPrediction({
      id: 2,
      market: 'us',
      symbol: 'TSLA',
      percentile_rank: 0.42,
    })
    expect(result.percentileRank).toBe(42)
  })

  it('infers direction "up" when up_probability > 0.55', () => {
    const result = transformPrediction({
      id: 3,
      market: 'us',
      symbol: 'MSFT',
      up_probability: 0.7,
    })
    expect(result.predictedDirection).toBe('up')
  })

  it('infers direction "down" when up_probability < 0.45', () => {
    const result = transformPrediction({
      id: 4,
      market: 'us',
      symbol: 'META',
      up_probability: 0.3,
    })
    expect(result.predictedDirection).toBe('down')
  })

  it('infers direction "neutral" when up_probability is between 0.45 and 0.55', () => {
    const result = transformPrediction({
      id: 5,
      market: 'us',
      symbol: 'GOOG',
      up_probability: 0.5,
    })
    expect(result.predictedDirection).toBe('neutral')
  })

  it('uses predicted_direction over inferred direction when provided', () => {
    const result = transformPrediction({
      id: 6,
      market: 'us',
      symbol: 'AMZN',
      up_probability: 0.7, // would infer "up"
      predicted_direction: 'down', // explicit override
    })
    expect(result.predictedDirection).toBe('down')
  })

  it('uses defaults when fields are missing', () => {
    const result = transformPrediction({
      id: 7,
      market: 'cn',
      symbol: '600000',
    })

    expect(result.predictionDate).toBe('')
    expect(result.predictedScore).toBe(0)
    expect(result.percentileRank).toBe(0)
    expect(result.actualReturn).toBeNull()
    expect(result.forwardDays).toBe(5)
  })

  it('handles boundary up_probability of exactly 0.55 as neutral', () => {
    const result = transformPrediction({
      id: 8,
      market: 'us',
      symbol: 'NVDA',
      up_probability: 0.55,
    })
    expect(result.predictedDirection).toBe('neutral')
  })

  it('handles boundary up_probability of exactly 0.45 as neutral', () => {
    const result = transformPrediction({
      id: 9,
      market: 'us',
      symbol: 'AMD',
      up_probability: 0.45,
    })
    expect(result.predictedDirection).toBe('neutral')
  })
})
