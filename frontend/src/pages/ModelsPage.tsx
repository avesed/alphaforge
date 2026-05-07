import { useState, Fragment } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  History,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Loader2,
  BarChart3,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { getErrorMessage } from '@/api/client'
import { predictionsApi } from '@/api/predictions'
import type { FeatureImportance } from '@/api/predictions'
import { cn } from '@/lib/utils'
import type { Market, PredictionModel } from '@/types'

const MARKETS: Market[] = ['cn', 'us', 'hk']

const MARKET_LABELS: Record<Market, string> = {
  cn: 'CN',
  us: 'US',
  hk: 'HK',
}

function MarketFilter({
  value,
  onChange,
}: {
  value: Market | 'all'
  onChange: (m: Market | 'all') => void
}) {
  const options: Array<Market | 'all'> = ['all', ...MARKETS]
  return (
    <div className="flex gap-1 rounded-lg border bg-muted p-1">
      {options.map((m) => (
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
          {m === 'all' ? 'All' : m.toUpperCase()}
        </button>
      ))}
    </div>
  )
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

function MarketBadge({ market }: { market: Market }) {
  const colors: Record<Market, string> = {
    cn: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300',
    us: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300',
    hk: 'bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300',
  }
  return (
    <span
      className={cn(
        'inline-flex rounded-full px-2 py-0.5 text-xs font-medium',
        colors[market]
      )}
    >
      {MARKET_LABELS[market]}
    </span>
  )
}

function FeatureImportancePanel({ modelId }: { modelId: number }) {
  const featureQuery = useQuery({
    queryKey: ['featureImportance', modelId],
    queryFn: () =>
      predictionsApi
        .getFeatureImportance(String(modelId))
        .then((r) => r.data),
  })

  if (featureQuery.isLoading) {
    return (
      <div className="space-y-2 p-4">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} className="h-6 w-full" />
        ))}
      </div>
    )
  }

  if (featureQuery.isError) {
    return (
      <div className="p-4 text-sm text-destructive">
        Failed to load feature importance.{' '}
        <button className="underline" onClick={() => void featureQuery.refetch()}>
          Retry
        </button>
      </div>
    )
  }

  const features: FeatureImportance[] = featureQuery.data ?? []
  const topFeatures = features.slice(0, 15)
  const maxImportance =
    topFeatures.length > 0
      ? Math.max(...topFeatures.map((f) => f.importance))
      : 1

  if (topFeatures.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        No feature importance data available.
      </div>
    )
  }

  return (
    <div className="space-y-2 p-4">
      <h4 className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
        <BarChart3 className="h-4 w-4" />
        Top {topFeatures.length} Features
      </h4>
      <div className="space-y-1.5">
        {topFeatures.map((f) => {
          const pct =
            maxImportance > 0 ? (f.importance / maxImportance) * 100 : 0
          return (
            <div key={f.feature} className="flex items-center gap-3 text-xs">
              <span className="w-40 truncate font-mono text-muted-foreground">
                {f.feature}
              </span>
              <div className="flex-1">
                <div className="h-4 rounded-sm bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-sm bg-primary/70 transition-all duration-300"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
              <span className="w-16 text-right font-mono">
                {f.importance.toFixed(1)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function ModelsPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [marketFilter, setMarketFilter] = useState<Market | 'all'>('all')
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const modelsQuery = useQuery({
    queryKey: ['models', marketFilter === 'all' ? undefined : marketFilter],
    queryFn: () =>
      predictionsApi
        .getModels(marketFilter === 'all' ? undefined : marketFilter)
        .then((r) => r.data),
  })

  const qualityMutation = useMutation({
    mutationFn: ({
      modelId,
      passed,
    }: {
      modelId: number
      passed: boolean
    }) => predictionsApi.updateModelQuality(String(modelId), passed),
    onSuccess: () => {
      toast({ title: 'Quality updated' })
      void queryClient.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (err) => {
      toast({
        title: 'Update failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const models: PredictionModel[] = modelsQuery.data ?? []
  const sortedModels = [...models].sort(
    (a, b) => b.modelDate.localeCompare(a.modelDate)
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            <History className="mr-2 inline-block h-6 w-6" />
            {t('nav.models')}
          </h2>
          <p className="text-muted-foreground">
            Model history, quality control, and feature importance
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void modelsQuery.refetch()}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          {t('common.refresh')}
        </Button>
      </div>

      <MarketFilter value={marketFilter} onChange={setMarketFilter} />

      {modelsQuery.isLoading && (
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3">
              {Array.from({ length: 6 }, (_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {modelsQuery.isError && (
        <Card>
          <CardContent className="p-6">
            <div className="text-center text-sm text-destructive">
              Failed to load models.{' '}
              <button
                className="underline"
                onClick={() => void modelsQuery.refetch()}
              >
                Retry
              </button>
            </div>
          </CardContent>
        </Card>
      )}

      {modelsQuery.isSuccess && sortedModels.length === 0 && (
        <Card>
          <CardContent className="p-12 text-center text-muted-foreground">
            No models found. Run a prediction to train the first model.
          </CardContent>
        </Card>
      )}

      {sortedModels.length > 0 && (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="p-4 pr-2 font-medium w-8"></th>
                    <th className="p-4 font-medium">Date</th>
                    <th className="p-4 font-medium">Market</th>
                    <th className="p-4 font-medium text-right">IC</th>
                    <th className="p-4 font-medium text-right">ICIR</th>
                    <th className="p-4 font-medium text-right">NDCG</th>
                    <th className="p-4 font-medium">Quality</th>
                    <th className="p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedModels.map((model) => {
                    const isExpanded = expandedId === model.id
                    return (
                      <Fragment key={model.id}>
                        <tr
                          className={cn(
                            'border-b border-border/50 cursor-pointer transition-colors hover:bg-muted/50',
                            isExpanded && 'bg-muted/30'
                          )}
                          onClick={() =>
                            setExpandedId(isExpanded ? null : model.id)
                          }
                        >
                          <td className="p-4 pr-2">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-4 w-4 text-muted-foreground" />
                            )}
                          </td>
                          <td className="p-4 font-mono">{model.modelDate}</td>
                          <td className="p-4">
                            <MarketBadge market={model.market} />
                          </td>
                          <td
                            className={cn(
                              'p-4 text-right font-mono font-medium',
                              model.ic > 0.02 &&
                                'text-green-600 dark:text-green-400',
                              model.ic >= 0.01 &&
                                model.ic <= 0.02 &&
                                'text-yellow-600 dark:text-yellow-400',
                              model.ic < 0.01 &&
                                'text-red-600 dark:text-red-400'
                            )}
                          >
                            {model.ic.toFixed(4)}
                          </td>
                          <td className="p-4 text-right font-mono">
                            {model.icir.toFixed(4)}
                          </td>
                          <td className="p-4 text-right font-mono">
                            {model.ndcg.toFixed(4)}
                          </td>
                          <td className="p-4">
                            <QualityBadge passed={model.qualityPassed} />
                          </td>
                          <td className="p-4">
                            <div
                              className="flex gap-1"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {model.qualityPassed ? (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7 text-xs text-destructive hover:text-destructive"
                                  disabled={qualityMutation.isPending}
                                  onClick={() =>
                                    qualityMutation.mutate({
                                      modelId: model.id,
                                      passed: false,
                                    })
                                  }
                                >
                                  {qualityMutation.isPending &&
                                  qualityMutation.variables?.modelId ===
                                    model.id ? (
                                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  ) : (
                                    <XCircle className="mr-1 h-3 w-3" />
                                  )}
                                  Reject
                                </Button>
                              ) : (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7 text-xs text-green-600 hover:text-green-600"
                                  disabled={qualityMutation.isPending}
                                  onClick={() =>
                                    qualityMutation.mutate({
                                      modelId: model.id,
                                      passed: true,
                                    })
                                  }
                                >
                                  {qualityMutation.isPending &&
                                  qualityMutation.variables?.modelId ===
                                    model.id ? (
                                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  ) : (
                                    <CheckCircle2 className="mr-1 h-3 w-3" />
                                  )}
                                  Approve
                                </Button>
                              )}
                            </div>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={8} className="bg-muted/20">
                              <FeatureImportancePanel modelId={model.id} />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
