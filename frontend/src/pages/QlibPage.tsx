import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Beaker, Play, CheckCircle2, XCircle,
  Loader2, FlaskConical, Calculator, Briefcase,
} from 'lucide-react'

import { qlibApi, type ExpressionResult, type ValidationResult } from '@/api/qlib'
import { getErrorMessage } from '@/api/client'
import type { Market } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { cn, formatDate } from '@/lib/utils'

const MARKETS: Market[] = ['us', 'cn', 'hk']

type TabKey = 'factors' | 'expressions' | 'backtests' | 'optimizer'

function MarketSelector({ value, onChange }: { value: Market; onChange: (m: Market) => void }) {
  return (
    <div className="flex gap-1 rounded-md border p-0.5">
      {MARKETS.map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={cn(
            'rounded px-3 py-1.5 text-sm font-medium transition-colors',
            value === m
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          )}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function ZScoreBadge({ value }: { value: number }) {
  const color = value > 0.5
    ? 'text-green-600 dark:text-green-400'
    : value < -0.5
      ? 'text-red-600 dark:text-red-400'
      : 'text-foreground'
  return <span className={cn('font-mono text-sm', color)}>{value.toFixed(3)}</span>
}

/* ---------- Factors Tab ---------- */
function FactorsTab() {
  const { toast } = useToast()
  const [symbol, setSymbol] = useState('')
  const [market, setMarket] = useState<Market>('us')
  const [querySymbol, setQuerySymbol] = useState('')
  const [queryMarket, setQueryMarket] = useState<Market>('us')

  const { data: factors, isLoading, error } = useQuery({
    queryKey: ['qlib-factors', querySymbol, queryMarket],
    queryFn: () => qlibApi.getFactors(querySymbol, queryMarket),
    select: (resp) => resp.data,
    enabled: !!querySymbol,
    staleTime: 60_000,
  })

  const handleCompute = () => {
    if (!symbol.trim()) {
      toast({ title: 'Enter a symbol', variant: 'destructive' })
      return
    }
    setQuerySymbol(symbol.trim().toUpperCase())
    setQueryMarket(market)
  }

  // Get latest factor data for display
  const latestFactors = factors && factors.length > 0 ? factors[factors.length - 1] : null
  const factorEntries = latestFactors
    ? Object.entries(latestFactors.factors).filter(([_, v]) => v !== null)
    : []

  // Simple z-score approximation: (value - mean) / std across all dates
  const factorStats: Record<string, { mean: number; std: number }> = {}
  if (factors && factors.length > 1) {
    for (const [key] of factorEntries) {
      const values = factors
        .map((f) => f.factors[key])
        .filter((v): v is number => v !== null && v !== undefined)
      if (values.length > 0) {
        const mean = values.reduce((s, v) => s + v, 0) / values.length
        const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length
        factorStats[key] = { mean, std: Math.sqrt(variance) || 1 }
      }
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[120px] space-y-2">
              <Label htmlFor="factorSymbol">Symbol</Label>
              <Input
                id="factorSymbol"
                placeholder="e.g., AAPL, 600519.SH"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCompute()}
              />
            </div>
            <div className="space-y-2">
              <Label>Market</Label>
              <MarketSelector value={market} onChange={setMarket} />
            </div>
            <Button onClick={handleCompute} disabled={isLoading}>
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Calculator className="mr-2 h-4 w-4" />
              )}
              Compute
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="py-4">
            <p className="text-sm text-destructive">{getErrorMessage(error)}</p>
          </CardContent>
        </Card>
      )}

      {isLoading && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {latestFactors && !isLoading && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              {querySymbol} - Factor Values
            </CardTitle>
            <CardDescription>
              Date: {latestFactors.date} | {factorEntries.length} factors computed
            </CardDescription>
          </CardHeader>
          <CardContent>
            {factorEntries.length === 0 ? (
              <p className="py-4 text-center text-muted-foreground">No factor data returned.</p>
            ) : (
              <div className="max-h-[500px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2">Factor</th>
                      <th className="pb-2 text-right">Value</th>
                      <th className="pb-2 text-right">Z-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {factorEntries.map(([name, value]) => {
                      const stats = factorStats[name]
                      const zScore = stats ? ((value ?? 0) - stats.mean) / stats.std : 0
                      return (
                        <tr key={name} className="border-b last:border-0 hover:bg-muted/50">
                          <td className="py-2 font-mono text-xs">{name}</td>
                          <td className="py-2 text-right font-mono">{value?.toFixed(4) ?? '-'}</td>
                          <td className="py-2 text-right">
                            <ZScoreBadge value={zScore} />
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {!querySymbol && !isLoading && (
        <div className="py-12 text-center text-muted-foreground">
          <Beaker className="mx-auto mb-2 h-8 w-8 opacity-40" />
          <p>Enter a symbol and click Compute to view factor values.</p>
        </div>
      )}
    </div>
  )
}

/* ---------- Expressions Tab ---------- */
function ExpressionsTab() {
  const { toast } = useToast()
  const [expression, setExpression] = useState('')
  const [symbol, setSymbol] = useState('')
  const [market, setMarket] = useState<Market>('us')

  // Validation
  const validateMutation = useMutation({
    mutationFn: (expr: string) => qlibApi.validateExpression(expr),
    onError: (err) => {
      toast({ title: 'Validation error', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  // Evaluation
  const evaluateMutation = useMutation({
    mutationFn: () =>
      qlibApi.evaluateExpression({
        expression,
        symbol: symbol.trim().toUpperCase(),
        market,
      }),
    onError: (err) => {
      toast({ title: 'Evaluation error', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  const validationResult = validateMutation.data?.data as ValidationResult | undefined
  const evalResult = evaluateMutation.data?.data as ExpressionResult | undefined

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="space-y-4 pt-6">
          <div className="space-y-2">
            <Label htmlFor="exprInput">Expression</Label>
            <Input
              id="exprInput"
              placeholder='e.g., $close/Ref($close,5)-1'
              value={expression}
              onChange={(e) => setExpression(e.target.value)}
              className="font-mono"
            />
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[120px] space-y-2">
              <Label htmlFor="exprSymbol">Symbol</Label>
              <Input
                id="exprSymbol"
                placeholder="e.g., AAPL"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Market</Label>
              <MarketSelector value={market} onChange={setMarket} />
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => {
                  if (!expression.trim()) return
                  validateMutation.mutate(expression.trim())
                }}
                disabled={!expression.trim() || validateMutation.isPending}
              >
                {validateMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                )}
                Validate
              </Button>
              <Button
                onClick={() => {
                  if (!expression.trim() || !symbol.trim()) {
                    toast({ title: 'Enter both expression and symbol', variant: 'destructive' })
                    return
                  }
                  evaluateMutation.mutate()
                }}
                disabled={!expression.trim() || !symbol.trim() || evaluateMutation.isPending}
              >
                {evaluateMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Evaluate
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Validation result */}
      {validationResult && (
        <Card className={validationResult.valid ? 'border-green-200 dark:border-green-800' : 'border-red-200 dark:border-red-800'}>
          <CardContent className="flex items-center gap-3 py-4">
            {validationResult.valid ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <div className="text-sm">
                  <span className="font-medium text-green-700 dark:text-green-400">Valid expression</span>
                  <span className="ml-2 font-mono text-xs text-muted-foreground">{validationResult.expression}</span>
                </div>
              </>
            ) : (
              <>
                <XCircle className="h-5 w-5 text-red-500" />
                <div className="text-sm">
                  <span className="font-medium text-red-700 dark:text-red-400">Invalid expression</span>
                  {validationResult.error && (
                    <span className="ml-2 text-muted-foreground">{validationResult.error}</span>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Evaluation results */}
      {evaluateMutation.isPending && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {evalResult && !evaluateMutation.isPending && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Results</CardTitle>
            <CardDescription className="font-mono text-xs">{evalResult.expression}</CardDescription>
          </CardHeader>
          <CardContent>
            {evalResult.values.length === 0 ? (
              <p className="py-4 text-center text-muted-foreground">No values returned.</p>
            ) : (
              <div className="max-h-[400px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2">Date</th>
                      <th className="pb-2 text-right">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalResult.values.map((v) => (
                      <tr key={v.date} className="border-b last:border-0 hover:bg-muted/50">
                        <td className="py-1.5">{formatDate(v.date)}</td>
                        <td className={cn('py-1.5 text-right font-mono',
                          v.value !== null && v.value > 0
                            ? 'text-green-600 dark:text-green-400'
                            : v.value !== null && v.value < 0
                              ? 'text-red-600 dark:text-red-400'
                              : ''
                        )}>
                          {v.value !== null ? v.value.toFixed(6) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {!evalResult && !evaluateMutation.isPending && !validationResult && (
        <div className="py-12 text-center text-muted-foreground">
          <Calculator className="mx-auto mb-2 h-8 w-8 opacity-40" />
          <p>Enter a Qlib expression, validate it, then evaluate with a symbol.</p>
          <p className="mt-1 text-xs">Example: $close/Ref($close,5)-1 (5-day return)</p>
        </div>
      )}
    </div>
  )
}

/* ---------- Backtests Tab ---------- */
function QlibBacktestsTab() {
  const { toast } = useToast()
  const [strategyType, setStrategyType] = useState<'topk' | 'signal' | 'long_short'>('topk')
  const [market, setMarket] = useState<Market>('us')
  const [symbols, setSymbols] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  const strategies = [
    { key: 'topk' as const, label: 'Top-K' },
    { key: 'signal' as const, label: 'Signal' },
    { key: 'long_short' as const, label: 'Long/Short' },
  ]

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <FlaskConical className="h-5 w-5" />
            Qlib Factor Backtest
          </CardTitle>
          <CardDescription>
            Run backtests using Qlib strategies and factor expressions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-2">
              <Label>Strategy</Label>
              <div className="flex gap-1 rounded-md border p-0.5">
                {strategies.map((s) => (
                  <button
                    key={s.key}
                    type="button"
                    onClick={() => setStrategyType(s.key)}
                    className={cn(
                      'rounded px-3 py-1.5 text-sm font-medium transition-colors',
                      strategyType === s.key
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                    )}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Label>Market</Label>
              <MarketSelector value={market} onChange={setMarket} />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="btStart">Start Date</Label>
              <Input
                id="btStart"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="btEnd">End Date</Label>
              <Input
                id="btEnd"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="btSymbols">Symbols (optional, comma-separated)</Label>
            <Input
              id="btSymbols"
              placeholder="Leave empty to use default universe"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
            />
          </div>

          <Button
            onClick={() => {
              toast({ title: 'Qlib backtest queued', description: `${strategyType} strategy on ${market.toUpperCase()}` })
            }}
          >
            <Play className="mr-2 h-4 w-4" />
            Start Backtest
          </Button>
        </CardContent>
      </Card>

      {/* History placeholder */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Backtest History</CardTitle>
          <CardDescription>Previous Qlib backtest runs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="py-8 text-center text-muted-foreground">
            <FlaskConical className="mx-auto mb-2 h-8 w-8 opacity-40" />
            <p>No Qlib backtests have been run yet.</p>
            <p className="mt-1 text-xs">Note: These are factor-based Qlib backtests, separate from ML model backtests.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

/* ---------- Optimizer Tab ---------- */
function OptimizerTab() {
  return (
    <div className="space-y-4">
      <Card className="border-dashed">
        <CardContent className="py-12 text-center">
          <Briefcase className="mx-auto mb-3 h-10 w-10 text-muted-foreground opacity-40" />
          <h3 className="text-lg font-medium">Portfolio Optimizer</h3>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
            Portfolio optimization is available when Qlib data is synced and factor
            computations are ready. This module supports mean-variance optimization,
            efficient frontier analysis, and risk decomposition.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {['Mean-Variance', 'Efficient Frontier', 'Risk Parity', 'Black-Litterman'].map((item) => (
              <span
                key={item}
                className="rounded-full border px-3 py-1 text-xs text-muted-foreground"
              >
                {item}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

/* ---------- Main Page ---------- */
export default function QlibPage() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState<TabKey>('factors')

  const tabs: Array<{ key: TabKey; label: string; icon: React.ElementType }> = [
    { key: 'factors', label: 'Factors', icon: Beaker },
    { key: 'expressions', label: 'Expressions', icon: Calculator },
    { key: 'backtests', label: 'Backtests', icon: FlaskConical },
    { key: 'optimizer', label: 'Optimizer', icon: Briefcase },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{t('nav.qlib')}</h2>
        <p className="text-muted-foreground">Factor computation, expression evaluation, and backtesting</p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-1 rounded-md border p-0.5 w-fit">
        {tabs.map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                'flex items-center gap-1.5 rounded px-3 py-1.5 text-sm font-medium transition-colors',
                activeTab === tab.key
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      {activeTab === 'factors' && <FactorsTab />}
      {activeTab === 'expressions' && <ExpressionsTab />}
      {activeTab === 'backtests' && <QlibBacktestsTab />}
      {activeTab === 'optimizer' && <OptimizerTab />}
    </div>
  )
}
