import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  LayoutDashboard,
  Target,
  Activity,
  Wifi,
  WifiOff,
  CheckCircle2,
  XCircle,
  Play,
  Loader2,
  Server,
  Crown,
  RefreshCw,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { getErrorMessage } from '@/api/client'
import { statsApi } from '@/api/stats'
import { predictionsApi } from '@/api/predictions'
import { cn } from '@/lib/utils'
import type { Market, PredictionModel } from '@/types'

const MARKETS: Market[] = ['cn', 'us', 'hk']

const MARKET_LABELS: Record<Market, string> = {
  cn: 'China A-Share',
  us: 'US Market',
  hk: 'Hong Kong',
}

function formatIc(v: number | null | undefined): string {
  if (v == null) return '--'
  return v.toFixed(4)
}

function icColor(ic: number | null | undefined): string {
  if (ic == null) return ''
  if (ic > 0.02) return 'text-green-600 dark:text-green-400'
  if (ic >= 0.01) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

function QualityBadge({ passed }: { passed: boolean }) {
  if (passed) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900/50 dark:text-green-300">
        <CheckCircle2 className="h-3 w-3" />
        Pass
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/50 dark:text-red-300">
      <XCircle className="h-3 w-3" />
      Fail
    </span>
  )
}

function MarketStatusCard({ market, model }: { market: Market; model: PredictionModel | undefined }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            {MARKET_LABELS[market]}
          </span>
          {model != null && <QualityBadge passed={model.qualityPassed} />}
        </CardTitle>
        <CardDescription>
          {model != null ? `Model: ${model.modelDate}` : 'No model trained yet'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-muted-foreground">IC</span>
          <span className={cn('font-mono font-medium', icColor(model?.ic))}>
            {formatIc(model?.ic)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">ICIR</span>
          <span className={cn('font-mono font-medium', icColor(model?.icir))}>
            {formatIc(model?.icir)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">NDCG</span>
          <span className="font-mono">{formatIc(model?.ndcg)}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function MarketStatusSkeleton() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-4 w-24" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
      </CardContent>
    </Card>
  )
}

export default function DashboardPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const statsQuery = useQuery({
    queryKey: ['stats'],
    queryFn: () => statsApi.get().then((r) => r.data),
  })

  const modelsQuery = useQuery({
    queryKey: ['models', 'all'],
    queryFn: () => predictionsApi.getModels().then((r) => r.data),
  })

  const triggerMutation = useMutation({
    mutationFn: (market: Market) => predictionsApi.triggerPrediction(market),
    onSuccess: (_data, market) => {
      toast({
        title: 'Prediction triggered',
        description: `Started prediction for ${MARKET_LABELS[market]}`,
      })
      void queryClient.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (err) => {
      toast({
        title: 'Trigger failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  // Find latest ranking model per market (ranking models have IC/ICIR/NDCG)
  const latestModels: Record<Market, PredictionModel | undefined> = {
    cn: undefined,
    us: undefined,
    hk: undefined,
  }
  if (modelsQuery.data) {
    for (const model of modelsQuery.data) {
      const existing = latestModels[model.market]
      if (existing == null) {
        latestModels[model.market] = model
      } else if (model.modelDate > existing.modelDate) {
        latestModels[model.market] = model
      } else if (model.modelDate === existing.modelDate && model.modelType === 'ranking' && existing.modelType !== 'ranking') {
        latestModels[model.market] = model
      }
    }
  }

  const stats = statsQuery.data

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            <LayoutDashboard className="mr-2 inline-block h-6 w-6" />
            {t('dashboard.title')}
          </h2>
          <p className="text-muted-foreground">{t('dashboard.subtitle')}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void queryClient.invalidateQueries({ queryKey: ['stats'] })
            void queryClient.invalidateQueries({ queryKey: ['models'] })
          }}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          {t('common.refresh')}
        </Button>
      </div>

      {/* Market status cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {modelsQuery.isLoading
          ? MARKETS.map((m) => <MarketStatusSkeleton key={m} />)
          : MARKETS.map((m) => (
              <MarketStatusCard key={m} market={m} model={latestModels[m]} />
            ))}
      </div>

      {modelsQuery.isError && (
        <div className="rounded-md bg-destructive/10 p-4 text-sm text-destructive">
          Failed to load model data.{' '}
          <button
            className="underline"
            onClick={() => void modelsQuery.refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {/* Bottom row: Quick Actions + System Status */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Play className="h-4 w-4" />
              Quick Actions
            </CardTitle>
            <CardDescription>Trigger prediction runs for each market</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              {MARKETS.map((m) => (
                <Button
                  key={m}
                  variant="outline"
                  disabled={triggerMutation.isPending}
                  onClick={() => triggerMutation.mutate(m)}
                >
                  {triggerMutation.isPending &&
                  triggerMutation.variables === m ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  Run {MARKET_LABELS[m]}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4" />
              System Status
            </CardTitle>
            <CardDescription>Service connectivity overview</CardDescription>
          </CardHeader>
          <CardContent>
            {statsQuery.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-3/4" />
              </div>
            ) : statsQuery.isError ? (
              <div className="text-sm text-destructive">
                Failed to load stats.{' '}
                <button
                  className="underline"
                  onClick={() => void statsQuery.refetch()}
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Crown className="h-4 w-4" />
                    Scheduler
                  </span>
                  <span className="flex items-center gap-2">
                    {stats?.schedulerIsLeader ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900/50 dark:text-green-300">
                        Leader
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                        Follower
                      </span>
                    )}
                    <span className="text-muted-foreground">
                      {stats?.schedulerJobCount ?? 0} jobs
                    </span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Server className="h-4 w-4" />
                    StockPulse
                  </span>
                  {stats?.stockpulseConnected ? (
                    <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                      <Wifi className="h-4 w-4" />
                      Connected
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
                      <WifiOff className="h-4 w-4" />
                      Disconnected
                    </span>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Activity className="h-4 w-4" />
                    Redis
                  </span>
                  {stats?.redisConnected ? (
                    <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                      <Wifi className="h-4 w-4" />
                      Connected
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
                      <WifiOff className="h-4 w-4" />
                      Disconnected
                    </span>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
