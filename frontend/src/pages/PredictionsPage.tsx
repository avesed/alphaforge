import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Target,
  Play,
  Loader2,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  BarChart3,
  TrendingUp,
  Activity,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { getErrorMessage } from '@/api/client'
import { predictionsApi } from '@/api/predictions'
import { cn } from '@/lib/utils'
import type { Market, TaskStatus } from '@/types'

const MARKETS: Market[] = ['cn', 'us', 'hk']

const MARKET_LABELS: Record<Market, string> = {
  cn: 'China A-Share',
  us: 'US Market',
  hk: 'Hong Kong',
}

const DAYS_OPTIONS = [7, 30, 90] as const

// ── Inline shared components ──────────────────

function MarketSelector({
  value,
  onChange,
}: {
  value: Market
  onChange: (m: Market) => void
}) {
  return (
    <div className="flex gap-1 rounded-lg border bg-muted p-1">
      {MARKETS.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={cn(
            'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
            value === m
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function DaysSelector({
  value,
  onChange,
}: {
  value: number
  onChange: (d: number) => void
}) {
  return (
    <div className="flex gap-1 rounded-lg border bg-muted p-1">
      {DAYS_OPTIONS.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          className={cn(
            'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
            value === d
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {d}d
        </button>
      ))}
    </div>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'rounded-md px-4 py-2 text-sm font-medium transition-colors',
        active
          ? 'bg-primary text-primary-foreground'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      )}
    >
      {children}
    </button>
  )
}

function StatusBadge({ status }: { status: TaskStatus['status'] }) {
  const config = {
    pending: {
      cls: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300',
      label: 'Pending',
    },
    running: {
      cls: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300',
      label: 'Running',
    },
    completed: {
      cls: 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300',
      label: 'Completed',
    },
    failed: {
      cls: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300',
      label: 'Failed',
    },
  }
  const c = config[status]
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
        c.cls
      )}
    >
      {c.label}
    </span>
  )
}

function DirectionBadge({ direction }: { direction: 'up' | 'down' | 'neutral' }) {
  if (direction === 'up') {
    return (
      <span className="inline-flex items-center gap-0.5 text-green-600 dark:text-green-400">
        <ArrowUpRight className="h-3.5 w-3.5" />
        Up
      </span>
    )
  }
  if (direction === 'down') {
    return (
      <span className="inline-flex items-center gap-0.5 text-red-600 dark:text-red-400">
        <ArrowDownRight className="h-3.5 w-3.5" />
        Down
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-0.5 text-muted-foreground">
      <Minus className="h-3.5 w-3.5" />
      Neutral
    </span>
  )
}

// ── Trigger Tab ──────────────────────────────

function TriggerTab() {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [activeTasks, setActiveTasks] = useState<Record<Market, string | null>>({
    cn: null,
    us: null,
    hk: null,
  })
  const [forceRetrain, setForceRetrain] = useState<Record<Market, boolean>>({
    cn: false,
    us: false,
    hk: false,
  })

  const triggerMutation = useMutation({
    mutationFn: ({ market, force }: { market: Market; force: boolean }) =>
      predictionsApi.triggerPrediction(market, force),
    onSuccess: (resp, { market }) => {
      const taskId = resp.data.taskId
      setActiveTasks((prev) => ({ ...prev, [market]: taskId }))
      toast({
        title: 'Prediction triggered',
        description: `Task ${taskId} started for ${MARKET_LABELS[market]}`,
      })
    },
    onError: (err) => {
      toast({
        title: 'Trigger failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  // Poll active tasks
  const pollableMarkets = MARKETS.filter((m) => activeTasks[m] != null)

  const taskStatusQueries = useQuery({
    queryKey: ['taskStatus', activeTasks],
    queryFn: async () => {
      const results: Partial<Record<Market, TaskStatus>> = {}
      for (const m of pollableMarkets) {
        const taskId = activeTasks[m]
        if (taskId != null) {
          const resp = await predictionsApi.getTaskStatus(taskId)
          results[m] = resp.data
        }
      }
      return results
    },
    enabled: pollableMarkets.length > 0,
    refetchInterval: 3000,
  })

  // Clear completed/failed tasks from polling
  const clearCompleted = useCallback(() => {
    if (taskStatusQueries.data == null) return
    const updates: Partial<Record<Market, string | null>> = {}
    let changed = false
    for (const m of MARKETS) {
      const status = taskStatusQueries.data[m]
      if (
        status != null &&
        (status.status === 'completed' || status.status === 'failed')
      ) {
        updates[m] = null
        changed = true
      }
    }
    if (changed) {
      setActiveTasks((prev) => ({ ...prev, ...updates }))
      void queryClient.invalidateQueries({ queryKey: ['models'] })
    }
  }, [taskStatusQueries.data, queryClient])

  useEffect(() => {
    const timer = setTimeout(clearCompleted, 5000)
    return () => clearTimeout(timer)
  }, [clearCompleted])

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {MARKETS.map((market) => {
        const taskId = activeTasks[market]
        const taskData = taskStatusQueries.data?.[market]
        const isRunning = taskId != null

        return (
          <Card key={market}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">{MARKET_LABELS[market]}</CardTitle>
              <CardDescription>
                {isRunning && taskData != null ? (
                  <StatusBadge status={taskData.status} />
                ) : (
                  'Ready to run'
                )}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {isRunning && taskData != null && taskData.progress != null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Progress</span>
                    <span>{Math.round(taskData.progress)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${taskData.progress}%` }}
                    />
                  </div>
                </div>
              )}

              {taskData?.error != null && (
                <p className="text-xs text-destructive">{taskData.error}</p>
              )}

              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={forceRetrain[market]}
                    onChange={(e) =>
                      setForceRetrain((prev) => ({
                        ...prev,
                        [market]: e.target.checked,
                      }))
                    }
                    className="rounded border-input"
                    disabled={isRunning}
                  />
                  Force Retrain
                </label>
              </div>

              <Button
                className="w-full"
                disabled={isRunning || triggerMutation.isPending}
                onClick={() =>
                  triggerMutation.mutate({
                    market,
                    force: forceRetrain[market],
                  })
                }
              >
                {isRunning ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                {isRunning ? 'Running...' : 'Run Prediction'}
              </Button>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

// ── Results Tab ──────────────────────────────

function ResultsTab() {
  const [market, setMarket] = useState<Market>('us')

  const latestQuery = useQuery({
    queryKey: ['predictions', 'latest', market],
    queryFn: () => predictionsApi.getLatest(market).then((r) => r.data),
  })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <MarketSelector value={market} onChange={setMarket} />
        <Button
          variant="outline"
          size="sm"
          onClick={() => void latestQuery.refetch()}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {latestQuery.isLoading && (
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3">
              {Array.from({ length: 8 }, (_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {latestQuery.isError && (
        <Card>
          <CardContent className="p-6">
            <div className="text-center text-sm text-destructive">
              Failed to load predictions.{' '}
              <button className="underline" onClick={() => void latestQuery.refetch()}>
                Retry
              </button>
            </div>
          </CardContent>
        </Card>
      )}

      {latestQuery.data != null && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">
                  {MARKET_LABELS[market]} Predictions
                </CardTitle>
                <CardDescription>
                  {latestQuery.data.predictionDate ?? 'No date'}
                  {' | '}{latestQuery.data.count} predictions
                </CardDescription>
              </div>
              <span className="text-sm text-muted-foreground">
                {latestQuery.data.predictions.length} symbols
              </span>
            </div>
          </CardHeader>
          <CardContent>
            {latestQuery.data.predictions.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No predictions available for this market.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2 pr-4 font-medium">Symbol</th>
                      <th className="pb-2 pr-4 font-medium text-right">Score</th>
                      <th className="pb-2 pr-4 font-medium text-right">Percentile</th>
                      <th className="pb-2 pr-4 font-medium">Direction</th>
                      <th className="pb-2 font-medium text-right">Actual Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {latestQuery.data.predictions.map((pred) => (
                      <tr
                        key={pred.id}
                        className="border-b border-border/50 last:border-0"
                      >
                        <td className="py-2 pr-4 font-mono font-medium">
                          {pred.symbol}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono">
                          {pred.predictedScore.toFixed(4)}
                        </td>
                        <td className="py-2 pr-4 text-right">
                          <span
                            className={cn(
                              'font-medium',
                              pred.percentileRank >= 80 &&
                                'text-green-600 dark:text-green-400',
                              pred.percentileRank <= 20 &&
                                'text-red-600 dark:text-red-400'
                            )}
                          >
                            {pred.percentileRank.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-2 pr-4">
                          <DirectionBadge direction={pred.predictedDirection} />
                        </td>
                        <td className="py-2 text-right font-mono">
                          {pred.actualReturn != null ? (
                            <span
                              className={cn(
                                pred.actualReturn > 0
                                  ? 'text-green-600 dark:text-green-400'
                                  : pred.actualReturn < 0
                                    ? 'text-red-600 dark:text-red-400'
                                    : ''
                              )}
                            >
                              {pred.actualReturn > 0 ? '+' : ''}
                              {(pred.actualReturn * 100).toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-muted-foreground">--</span>
                          )}
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
    </div>
  )
}

// ── Accuracy Tab ──────────────────────────────

function AccuracyTab() {
  const [market, setMarket] = useState<Market>('us')
  const [days, setDays] = useState<number>(30)

  const accuracyQuery = useQuery({
    queryKey: ['predictions', 'accuracy', market, days],
    queryFn: () => predictionsApi.getAccuracy(market, days).then((r) => r.data),
  })

  const data = accuracyQuery.data

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <MarketSelector value={market} onChange={setMarket} />
        <DaysSelector value={days} onChange={setDays} />
      </div>

      {accuracyQuery.isLoading && (
        <div className="grid gap-4 md:grid-cols-3">
          {Array.from({ length: 3 }, (_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <Skeleton className="mb-2 h-4 w-24" />
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {accuracyQuery.isError && (
        <Card>
          <CardContent className="p-6">
            <div className="text-center text-sm text-destructive">
              Failed to load accuracy data.{' '}
              <button
                className="underline"
                onClick={() => void accuracyQuery.refetch()}
              >
                Retry
              </button>
            </div>
          </CardContent>
        </Card>
      )}

      {data != null && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Target className="h-4 w-4" />
                Direction Accuracy
              </div>
              <div
                className={cn(
                  'mt-2 text-3xl font-bold',
                  data.directionAccuracy != null && data.directionAccuracy >= 0.55
                    ? 'text-green-600 dark:text-green-400'
                    : data.directionAccuracy != null && data.directionAccuracy < 0.5
                      ? 'text-red-600 dark:text-red-400'
                      : ''
                )}
              >
                {data.directionAccuracy != null
                  ? `${(data.directionAccuracy * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {data.totalPredictions} predictions
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <TrendingUp className="h-4 w-4" />
                Average IC
              </div>
              <div
                className={cn(
                  'mt-2 text-3xl font-bold font-mono',
                  data.ic != null && data.ic > 0.02
                    ? 'text-green-600 dark:text-green-400'
                    : data.ic != null && data.ic < 0.01
                      ? 'text-red-600 dark:text-red-400'
                      : data.ic != null
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : ''
                )}
              >
                {data.ic != null ? data.ic.toFixed(4) : '--'}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <BarChart3 className="h-4 w-4" />
                Average ICIR
              </div>
              <div
                className={cn(
                  'mt-2 text-3xl font-bold font-mono',
                  data.icir != null && data.icir > 0.3
                    ? 'text-green-600 dark:text-green-400'
                    : data.icir != null && data.icir < 0.1
                      ? 'text-red-600 dark:text-red-400'
                      : data.icir != null
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : ''
                )}
              >
                {data.icir != null ? data.icir.toFixed(4) : '--'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

// ── Main Page ──────────────────────────────

export default function PredictionsPage() {
  const { t } = useTranslation()
  const [tab, setTab] = useState<'trigger' | 'results' | 'accuracy'>('trigger')

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">
          <Target className="mr-2 inline-block h-6 w-6" />
          {t('nav.predictions')}
        </h2>
        <p className="text-muted-foreground">
          Trigger, view results, and track prediction accuracy
        </p>
      </div>

      <div className="flex gap-1 rounded-lg border bg-muted p-1">
        <TabButton active={tab === 'trigger'} onClick={() => setTab('trigger')}>
          <Play className="mr-1.5 inline h-4 w-4" />
          Trigger
        </TabButton>
        <TabButton active={tab === 'results'} onClick={() => setTab('results')}>
          <Activity className="mr-1.5 inline h-4 w-4" />
          Results
        </TabButton>
        <TabButton
          active={tab === 'accuracy'}
          onClick={() => setTab('accuracy')}
        >
          <BarChart3 className="mr-1.5 inline h-4 w-4" />
          Accuracy
        </TabButton>
      </div>

      {tab === 'trigger' && <TriggerTab />}
      {tab === 'results' && <ResultsTab />}
      {tab === 'accuracy' && <AccuracyTab />}
    </div>
  )
}
