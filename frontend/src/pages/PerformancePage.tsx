import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  BarChart3, TrendingDown, Activity,
  Target, Layers, ArrowDownUp,
} from 'lucide-react'

import { predictionsApi } from '@/api/predictions'
import type { Market } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn, formatDate } from '@/lib/utils'

const MARKETS: Market[] = ['us', 'cn', 'hk']
const DAY_OPTIONS = [30, 90, 180] as const

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

function DaysSelector({ value, onChange }: { value: number; onChange: (d: number) => void }) {
  return (
    <div className="flex gap-1 rounded-md border p-0.5">
      {DAY_OPTIONS.map((d) => (
        <button
          key={d}
          type="button"
          onClick={() => onChange(d)}
          className={cn(
            'rounded px-2.5 py-1 text-xs font-medium transition-colors',
            value === d
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          )}
        >
          {d}d
        </button>
      ))}
    </div>
  )
}

function SummaryCard({ title, value, subtitle, icon: Icon, valueColor }: {
  title: string
  value: string
  subtitle?: string | undefined
  icon: React.ElementType
  valueColor?: string | undefined
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className={cn('text-2xl font-bold', valueColor)}>{value}</div>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </CardContent>
    </Card>
  )
}

function IcBar({ value, maxAbs }: { value: number; maxAbs: number }) {
  const width = maxAbs > 0 ? Math.abs(value) / maxAbs * 100 : 0
  const isPositive = value >= 0
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-3 w-24 rounded bg-muted">
        <div
          className={cn(
            'absolute top-0 h-3 rounded transition-all',
            isPositive ? 'left-1/2 bg-green-500' : 'right-1/2 bg-red-500'
          )}
          style={{ width: `${width / 2}%` }}
        />
        <div className="absolute left-1/2 top-0 h-3 w-px bg-border" />
      </div>
      <span className={cn(
        'font-mono text-xs',
        isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
      )}>
        {value.toFixed(4)}
      </span>
    </div>
  )
}

/* ---------- Metrics Tab ---------- */
function MetricsTab({ market, days }: { market: Market; days: number }) {
  const { data: performance, isLoading } = useQuery({
    queryKey: ['performance', market, days],
    queryFn: () => predictionsApi.getPerformance(market, days),
    select: (resp) => resp.data,
    staleTime: 60_000,
  })

  const { data: accuracy } = useQuery({
    queryKey: ['accuracy', market, days],
    queryFn: () => predictionsApi.getAccuracy(market, days),
    select: (resp) => resp.data,
    staleTime: 60_000,
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-64" />
      </div>
    )
  }

  const metrics = performance ?? []
  const avgIc = metrics.length > 0
    ? metrics.reduce((sum, m) => sum + m.ic, 0) / metrics.length
    : 0
  const avgHitRate = accuracy?.overall.hitRate ?? 0
  const avgSpread = metrics.length > 0
    ? metrics.reduce((sum, m) => sum + m.spread, 0) / metrics.length
    : 0

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryCard
          title="Avg IC"
          value={avgIc.toFixed(4)}
          icon={Activity}
          valueColor={avgIc > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        />
        <SummaryCard
          title="Hit Rate"
          value={`${(avgHitRate * 100).toFixed(1)}%`}
          icon={Target}
          valueColor={avgHitRate > 0.5 ? 'text-green-600 dark:text-green-400' : undefined}
        />
        <SummaryCard
          title="Avg Spread"
          value={`${(avgSpread * 100).toFixed(2)}%`}
          icon={ArrowDownUp}
          valueColor={avgSpread > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        />
        <SummaryCard
          title="Total Dates"
          value={String(metrics.length)}
          subtitle={`${days}-day window`}
          icon={BarChart3}
        />
      </div>

      {/* Per-date table */}
      <Card>
        <CardContent className="pt-6">
          {metrics.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              No performance data available for this period.
            </div>
          ) : (
            <div className="max-h-[500px] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card">
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2">Date</th>
                    <th className="pb-2 text-right">IC</th>
                    <th className="pb-2 text-right">Hit Rate</th>
                    <th className="pb-2 text-right">Top-10 Ret</th>
                    <th className="pb-2 text-right">Bot-10 Ret</th>
                    <th className="pb-2 text-right">Spread</th>
                    <th className="pb-2 text-right">Symbols</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m) => (
                    <tr key={m.date} className="border-b last:border-0 hover:bg-muted/50">
                      <td className="py-2">{formatDate(m.date)}</td>
                      <td className={cn('py-2 text-right font-mono',
                        m.ic > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      )}>
                        {m.ic.toFixed(4)}
                      </td>
                      <td className="py-2 text-right font-mono">{(m.hitRate * 100).toFixed(1)}%</td>
                      <td className={cn('py-2 text-right font-mono',
                        m.topReturn > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      )}>
                        {(m.topReturn * 100).toFixed(2)}%
                      </td>
                      <td className={cn('py-2 text-right font-mono',
                        m.bottomReturn > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      )}>
                        {(m.bottomReturn * 100).toFixed(2)}%
                      </td>
                      <td className={cn('py-2 text-right font-mono',
                        m.spread > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      )}>
                        {(m.spread * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 text-right text-muted-foreground">{m.symbolCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

/* ---------- Signal Quality Tab ---------- */
function SignalQualityTab({ market, days }: { market: Market; days: number }) {
  const { data: icDecay, isLoading: icLoading } = useQuery({
    queryKey: ['ic-decay', market, days],
    queryFn: () => predictionsApi.getIcDecay(market, days),
    select: (resp) => resp.data,
    staleTime: 60_000,
  })

  const { data: turnover, isLoading: turnLoading } = useQuery({
    queryKey: ['turnover', market, days],
    queryFn: () => predictionsApi.getTurnover(market, days),
    select: (resp) => resp.data,
    staleTime: 60_000,
  })

  const isLoading = icLoading || turnLoading

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-48" />
        <Skeleton className="h-48" />
      </div>
    )
  }

  const icValues = icDecay?.icValues ?? []
  const icDays = icDecay?.days ?? []
  const maxAbsIc = icValues.length > 0 ? Math.max(...icValues.map(Math.abs), 0.001) : 0.01

  return (
    <div className="space-y-4">
      {/* IC Decay */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <TrendingDown className="h-5 w-5" />
            IC Decay Curve
          </CardTitle>
          <CardDescription>How predictive power decays over forward horizons</CardDescription>
        </CardHeader>
        <CardContent>
          {icValues.length === 0 ? (
            <p className="py-4 text-center text-muted-foreground">No IC decay data available.</p>
          ) : (
            <div className="space-y-2">
              {icDays.map((d, i) => {
                const ic = icValues[i] ?? 0
                return (
                  <div key={d} className="flex items-center gap-3">
                    <span className="w-10 text-right text-xs text-muted-foreground">t+{d}</span>
                    <IcBar value={ic} maxAbs={maxAbsIc} />
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Turnover */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <ArrowDownUp className="h-5 w-5" />
            Turnover
          </CardTitle>
          <CardDescription>Prediction stability over time</CardDescription>
        </CardHeader>
        <CardContent>
          {!turnover || turnover.turnoverRates.length === 0 ? (
            <p className="py-4 text-center text-muted-foreground">No turnover data available.</p>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="text-sm text-muted-foreground">Average Turnover</div>
                <div className="text-lg font-bold">{(turnover.avgTurnover * 100).toFixed(1)}%</div>
              </div>
              <div className="max-h-[300px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2">Date</th>
                      <th className="pb-2 text-right">Turnover Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {turnover.dates.map((date, i) => {
                      const rate = turnover.turnoverRates[i] ?? 0
                      return (
                        <tr key={date} className="border-b last:border-0">
                          <td className="py-1.5">{formatDate(date)}</td>
                          <td className="py-1.5 text-right font-mono">{(rate * 100).toFixed(1)}%</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

/* ---------- Attribution Tab ---------- */
function AttributionTab({ market, days }: { market: Market; days: number }) {
  const { data: attribution, isLoading } = useQuery({
    queryKey: ['attribution', market, days],
    queryFn: () => predictionsApi.getAttribution(market, days),
    select: (resp) => resp.data,
    staleTime: 60_000,
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-48" />
      </div>
    )
  }

  const categories = attribution?.categories ?? []
  const topFeatures = attribution?.topFeatures ?? []
  const totalContrib = categories.reduce((s, c) => s + Math.abs(c.contribution), 0) || 1

  return (
    <div className="space-y-4">
      {/* Category cards */}
      {categories.length === 0 ? (
        <div className="py-8 text-center text-muted-foreground">
          <Layers className="mx-auto mb-2 h-8 w-8 opacity-40" />
          <p>No attribution data available for this period.</p>
        </div>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            {categories.map((cat) => {
              const pct = (cat.contribution / totalContrib) * 100
              return (
                <Card key={cat.name}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium capitalize">{cat.name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className={cn('text-2xl font-bold',
                      cat.contribution > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    )}>
                      {pct.toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">{cat.featureCount} features</p>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Top features table */}
          {topFeatures.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Top Features</CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2">Rank</th>
                      <th className="pb-2">Feature</th>
                      <th className="pb-2 text-right">Importance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topFeatures.map((f) => (
                      <tr key={f.feature} className="border-b last:border-0 hover:bg-muted/50">
                        <td className="py-2 text-muted-foreground">#{f.rank}</td>
                        <td className="py-2 font-mono text-xs">{f.feature}</td>
                        <td className="py-2 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <div className="h-2 rounded-full bg-primary" style={{ width: `${Math.min(f.importance * 100, 100)}%`, minWidth: '4px' }} />
                            <span className="w-12 text-right font-mono text-xs">{(f.importance * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

/* ---------- Main Page ---------- */
export default function PerformancePage() {
  const { t } = useTranslation()
  const [market, setMarket] = useState<Market>('us')
  const [days, setDays] = useState(30)
  const [activeTab, setActiveTab] = useState<'metrics' | 'signal' | 'attribution'>('metrics')

  const tabs = [
    { key: 'metrics' as const, label: 'Metrics', icon: BarChart3 },
    { key: 'signal' as const, label: 'Signal Quality', icon: Activity },
    { key: 'attribution' as const, label: 'Attribution', icon: Layers },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{t('nav.performance')}</h2>
        <p className="text-muted-foreground">Prediction quality metrics and signal analysis</p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <MarketSelector value={market} onChange={setMarket} />
        <DaysSelector value={days} onChange={setDays} />
        <div className="flex gap-1 rounded-md border p-0.5">
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
      </div>

      {/* Tab content */}
      {activeTab === 'metrics' && <MetricsTab market={market} days={days} />}
      {activeTab === 'signal' && <SignalQualityTab market={market} days={days} />}
      {activeTab === 'attribution' && <AttributionTab market={market} days={days} />}
    </div>
  )
}
