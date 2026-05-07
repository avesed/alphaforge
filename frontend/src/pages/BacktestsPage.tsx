import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  FlaskConical, Play, Loader2, Trash2, ChevronDown, ChevronRight,
  Clock, CheckCircle2, XCircle, RotateCcw, Bot,
} from 'lucide-react'

import { backtestsApi, type BacktestConfig, type AgentBacktestParams } from '@/api/backtests'
import { getErrorMessage } from '@/api/client'
import type { Market } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { cn, formatDate } from '@/lib/utils'

type BacktestMode = 'static' | 'rolling' | 'agent'

const MARKETS: Market[] = ['us', 'cn', 'hk']
const TASK_KEY = 'alphaforge-backtest-task'

function getDefaultCutoff(): string {
  const d = new Date()
  d.setDate(d.getDate() - 60)
  return d.toISOString().split('T')[0] ?? ''
}

function formatDuration(startedAt?: string, completedAt?: string): string {
  if (!startedAt) return '-'
  const start = new Date(startedAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  const seconds = Math.round((end - start) / 1000)
  if (seconds < 60) return `${seconds}s`
  const min = Math.floor(seconds / 60)
  const sec = seconds % 60
  return `${min}m ${sec}s`
}

function formatIc(v: unknown): string {
  if (v === null || v === undefined || typeof v !== 'number') return '-'
  return v.toFixed(4)
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    completed: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  }
  const icons: Record<string, React.ReactNode> = {
    pending: <Clock className="h-3 w-3" />,
    running: <Loader2 className="h-3 w-3 animate-spin" />,
    completed: <CheckCircle2 className="h-3 w-3" />,
    failed: <XCircle className="h-3 w-3" />,
  }
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium', colors[status] ?? colors.pending)}>
      {icons[status]}
      {status}
    </span>
  )
}

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

export default function BacktestsPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const qc = useQueryClient()

  // Form state
  const [market, setMarket] = useState<Market>('us')
  const [mode, setMode] = useState<BacktestMode>('static')
  const [cutoffDate, setCutoffDate] = useState(getDefaultCutoff)
  const [validationDays, setValidationDays] = useState(10)
  const [forwardDays, setForwardDays] = useState(5)
  const [retrainInterval, setRetrainInterval] = useState(5)
  const [maxIterations, setMaxIterations] = useState(3)
  const [configOverride, setConfigOverride] = useState('')

  // Active task state, persisted to localStorage
  const [activeTaskId, setActiveTaskId] = useState<string | null>(() => {
    try {
      const stored = localStorage.getItem(TASK_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as { taskId: string; market: string }
        return parsed.taskId
      }
    } catch { /* ignore */ }
    return null
  })

  const persistTask = useCallback((taskId: string | null, mkt?: string) => {
    setActiveTaskId(taskId)
    if (taskId && mkt) {
      localStorage.setItem(TASK_KEY, JSON.stringify({ taskId, market: mkt }))
    } else {
      localStorage.removeItem(TASK_KEY)
    }
  }, [])

  // Expanded history row
  const [expandedId, setExpandedId] = useState<number | null>(null)

  // History query
  const { data: historyResp, isLoading: historyLoading } = useQuery({
    queryKey: ['backtests', market],
    queryFn: () => backtestsApi.list(market),
    select: (resp) => resp.data,
    staleTime: 30_000,
  })

  const history = historyResp?.backtests ?? []

  // Task polling
  const { data: taskStatus } = useQuery({
    queryKey: ['backtest-task', activeTaskId],
    queryFn: () => activeTaskId ? backtestsApi.getTaskStatus(activeTaskId) : null,
    select: (resp) => resp?.data ?? null,
    enabled: !!activeTaskId,
    refetchInterval: (query) => {
      const s = query.state.data?.data
      if (s && (s.status === 'completed' || s.status === 'failed')) return false
      return activeTaskId ? 3000 : false
    },
  })

  // Auto-clear and refresh on completion
  useEffect(() => {
    if (!taskStatus) return
    if (taskStatus.status === 'completed') {
      toast({ title: 'Backtest completed', description: 'Results are now available.' })
      qc.invalidateQueries({ queryKey: ['backtests'] })
      persistTask(null)
    } else if (taskStatus.status === 'failed') {
      toast({ title: 'Backtest failed', description: taskStatus.error ?? 'Unknown error', variant: 'destructive' })
      persistTask(null)
    }
  }, [taskStatus?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  // Start mutation
  const startMutation = useMutation({
    mutationFn: async () => {
      let override: Record<string, unknown> | undefined
      if (configOverride.trim()) {
        try {
          override = JSON.parse(configOverride) as Record<string, unknown>
        } catch {
          throw new Error('Invalid JSON in config override')
        }
      }

      if (mode === 'agent') {
        const params: AgentBacktestParams = {
          startDate: cutoffDate,
          maxRounds: maxIterations,
          ...(override ?? {}),
        }
        return backtestsApi.startAgentBacktest(market, params)
      }

      const config: BacktestConfig = {
        strategyType: mode === 'rolling' ? 'rolling' : 'static',
        startDate: cutoffDate,
        ...(mode === 'rolling' ? { retrainInterval } : {}),
        ...(override ?? {}),
      }
      return backtestsApi.startBacktest(market, config)
    },
    onSuccess: (resp) => {
      const data = resp.data
      persistTask(data.taskId, market)
      toast({ title: 'Backtest started' })
    },
    onError: (err) => {
      toast({ title: 'Failed to start backtest', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => backtestsApi.delete(id),
    onSuccess: () => {
      toast({ title: 'Backtest deleted' })
      qc.invalidateQueries({ queryKey: ['backtests'] })
    },
    onError: (err) => {
      toast({ title: 'Delete failed', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  const isRunning = !!activeTaskId && taskStatus?.status !== 'completed' && taskStatus?.status !== 'failed'

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{t('nav.backtests')}</h2>
        <p className="text-muted-foreground">Run static, rolling, or agent-driven backtests</p>
      </div>

      {/* Backtest Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <FlaskConical className="h-5 w-5" />
            New Backtest
          </CardTitle>
          <CardDescription>Configure and launch a backtest run</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Market + Mode */}
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-2">
              <Label>Market</Label>
              <MarketSelector value={market} onChange={setMarket} />
            </div>
            <div className="space-y-2">
              <Label>Mode</Label>
              <div className="flex gap-1 rounded-md border p-0.5">
                {(['static', 'rolling', 'agent'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={cn(
                      'rounded px-3 py-1.5 text-sm font-medium transition-colors capitalize',
                      mode === m
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                    )}
                  >
                    {m === 'agent' && <Bot className="mr-1 inline h-3.5 w-3.5" />}
                    {m}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Common fields */}
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="cutoff">Cutoff Date</Label>
              <Input
                id="cutoff"
                type="date"
                value={cutoffDate}
                onChange={(e) => setCutoffDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="valDays">Validation Days</Label>
              <Input
                id="valDays"
                type="number"
                min={5}
                max={250}
                value={validationDays}
                onChange={(e) => setValidationDays(Number(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="fwdDays">Forward Days</Label>
              <Input
                id="fwdDays"
                type="number"
                min={1}
                max={30}
                value={forwardDays}
                onChange={(e) => setForwardDays(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Mode-specific fields */}
          {mode === 'rolling' && (
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="retrain">Retrain Interval (days)</Label>
                <Input
                  id="retrain"
                  type="number"
                  min={1}
                  max={20}
                  value={retrainInterval}
                  onChange={(e) => setRetrainInterval(Number(e.target.value))}
                />
              </div>
            </div>
          )}

          {mode === 'agent' && (
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="maxIter">Max Iterations</Label>
                <Input
                  id="maxIter"
                  type="number"
                  min={1}
                  max={10}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                />
              </div>
            </div>
          )}

          {/* Config override */}
          <div className="space-y-2">
            <Label htmlFor="configOverride">Config Override (JSON, optional)</Label>
            <textarea
              id="configOverride"
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              placeholder='{"top_k": 30, "benchmark": "000300.SH"}'
              value={configOverride}
              onChange={(e) => setConfigOverride(e.target.value)}
            />
          </div>

          <Button
            onClick={() => startMutation.mutate()}
            disabled={startMutation.isPending || isRunning}
          >
            {startMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            {isRunning ? 'Backtest Running...' : 'Start Backtest'}
          </Button>
        </CardContent>
      </Card>

      {/* Active Task Progress */}
      {activeTaskId && taskStatus && (
        <Card className="border-blue-200 dark:border-blue-800">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-lg">
              <span className="flex items-center gap-2">
                <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                Active Backtest
              </span>
              <StatusBadge status={taskStatus.status} />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 text-sm sm:grid-cols-3">
              <div>
                <span className="text-muted-foreground">Task ID: </span>
                <span className="font-mono text-xs">{activeTaskId.slice(0, 12)}...</span>
              </div>
              <div>
                <span className="text-muted-foreground">Market: </span>
                <span className="font-medium uppercase">{taskStatus.market ?? market}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Elapsed: </span>
                <span className="font-medium">{formatDuration(taskStatus.startedAt)}</span>
              </div>
            </div>
            {taskStatus.progress !== undefined && taskStatus.progress > 0 && (
              <div className="mt-3">
                <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                  <span>Progress</span>
                  <span>{Math.round(taskStatus.progress * 100)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div
                    className="h-2 rounded-full bg-blue-500 transition-all"
                    style={{ width: `${Math.round(taskStatus.progress * 100)}%` }}
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* History Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-lg">
            <span>History</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => qc.invalidateQueries({ queryKey: ['backtests'] })}
            >
              <RotateCcw className="mr-1 h-4 w-4" />
              Refresh
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {historyLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : history.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              <FlaskConical className="mx-auto mb-2 h-8 w-8 opacity-40" />
              <p>No backtests yet. Start one above.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="w-8 pb-2" />
                    <th className="pb-2">Date</th>
                    <th className="pb-2">Market</th>
                    <th className="pb-2">Type</th>
                    <th className="pb-2">Status</th>
                    <th className="pb-2 text-right">IC</th>
                    <th className="pb-2 text-right">Duration</th>
                    <th className="w-10 pb-2" />
                  </tr>
                </thead>
                <tbody>
                  {history.map((bt) => {
                    const isExpanded = expandedId === bt.id
                    const results = bt.results ?? {}
                    return (
                      <tr key={bt.id} className="group">
                        <td colSpan={8} className="p-0">
                          <div
                            className="flex cursor-pointer items-center border-b px-0 py-2.5 transition-colors hover:bg-muted/50"
                            onClick={() => setExpandedId(isExpanded ? null : bt.id)}
                          >
                            <div className="w-8 text-muted-foreground">
                              {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            </div>
                            <div className="flex-1">{formatDate(bt.createdAt)}</div>
                            <div className="flex-1 font-medium uppercase">{bt.market}</div>
                            <div className="flex-1 capitalize">{bt.strategyType}</div>
                            <div className="flex-1"><StatusBadge status={bt.status} /></div>
                            <div className={cn('flex-1 text-right font-mono', bt.status === 'completed' && results.ic != null
                              ? (results.ic as number) > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                              : ''
                            )}>
                              {formatIc(results.ic)}
                            </div>
                            <div className="flex-1 text-right text-muted-foreground">
                              {formatDuration(bt.createdAt, bt.status === 'completed' ? bt.createdAt : undefined)}
                            </div>
                            <div className="w-10 text-right">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7 opacity-0 group-hover:opacity-100"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  if (window.confirm('Delete this backtest?')) {
                                    deleteMutation.mutate(String(bt.id))
                                  }
                                }}
                              >
                                <Trash2 className="h-3.5 w-3.5 text-destructive" />
                              </Button>
                            </div>
                          </div>
                          {isExpanded && bt.results && (
                            <div className="border-b bg-muted/30 px-8 py-4">
                              <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Detail Metrics</h4>
                              <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
                                {Object.entries(bt.results).map(([key, value]) => (
                                  <div key={key} className="space-y-0.5">
                                    <div className="text-xs text-muted-foreground">{key.replace(/_/g, ' ')}</div>
                                    <div className="font-mono text-sm">
                                      {typeof value === 'number' ? value.toFixed(4) : String(value ?? '-')}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
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
    </div>
  )
}
