import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Brain, Play, Square, Loader2, RefreshCw,
  XCircle, Clock,
  Zap, ToggleLeft, ToggleRight,
} from 'lucide-react'

import { rdagentApi } from '@/api/rdagent'
import { getErrorMessage } from '@/api/client'
import type { Market, RDAgentStatus } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { cn, formatDate } from '@/lib/utils'

const MARKETS: Market[] = ['us', 'cn', 'hk']

function StatusBadge({ status }: { status: RDAgentStatus['status'] }) {
  const styles: Record<string, string> = {
    idle: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
    starting: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    stopping: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
    error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  }
  const icons: Record<string, React.ReactNode> = {
    idle: <Clock className="h-3 w-3" />,
    starting: <Loader2 className="h-3 w-3 animate-spin" />,
    running: <Loader2 className="h-3 w-3 animate-spin" />,
    stopping: <Loader2 className="h-3 w-3 animate-spin" />,
    error: <XCircle className="h-3 w-3" />,
  }
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium', styles[status] ?? styles.idle)}>
      {icons[status]}
      {status}
    </span>
  )
}

function IcBadge({ value }: { value: number }) {
  const color = value >= 0.02
    ? 'text-green-600 dark:text-green-400'
    : value >= 0.01
      ? 'text-yellow-600 dark:text-yellow-400'
      : 'text-red-600 dark:text-red-400'
  return <span className={cn('font-mono', color)}>{value.toFixed(4)}</span>
}

function MarketControlCard({
  market,
  maxRoundsInput,
  setMaxRoundsInput,
}: {
  market: Market
  maxRoundsInput: number
  setMaxRoundsInput: (v: number) => void
}) {
  const { toast } = useToast()
  const qc = useQueryClient()

  const { data: status, isLoading } = useQuery({
    queryKey: ['rdagent-status', market],
    queryFn: () => rdagentApi.getStatus(market),
    select: (resp) => resp.data,
    refetchInterval: (query) => {
      const s = query.state.data?.data
      if (s && (s.status === 'running' || s.status === 'starting' || s.status === 'stopping')) {
        return 5000
      }
      return false
    },
  })

  const startMutation = useMutation({
    mutationFn: () => rdagentApi.start(market, maxRoundsInput),
    onSuccess: () => {
      toast({ title: `RD-Agent started for ${market.toUpperCase()}` })
      qc.invalidateQueries({ queryKey: ['rdagent-status', market] })
    },
    onError: (err) => {
      toast({ title: 'Failed to start', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  const stopMutation = useMutation({
    mutationFn: () => rdagentApi.stop(market),
    onSuccess: () => {
      toast({ title: `RD-Agent stopping for ${market.toUpperCase()}` })
      qc.invalidateQueries({ queryKey: ['rdagent-status', market] })
    },
    onError: (err) => {
      toast({ title: 'Failed to stop', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  const isActive = status?.status === 'running' || status?.status === 'starting' || status?.status === 'stopping'

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-24" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-10 w-full" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={cn(isActive && 'border-blue-200 dark:border-blue-800')}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{market.toUpperCase()}</CardTitle>
          <StatusBadge status={status?.status ?? 'idle'} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {status?.status === 'running' && (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Round</span>
              <span className="font-medium">{status.currentRound ?? 0} / {status.maxRounds ?? 30}</span>
            </div>
            {status.currentRound != null && status.maxRounds != null && (
              <div className="h-1.5 w-full rounded-full bg-muted">
                <div
                  className="h-1.5 rounded-full bg-blue-500 transition-all"
                  style={{ width: `${Math.round((status.currentRound / status.maxRounds) * 100)}%` }}
                />
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-muted-foreground">Factors found</span>
              <span className="font-medium">{status.factorsFound ?? 0}</span>
            </div>
          </div>
        )}

        {status?.status === 'error' && status.error && (
          <div className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            {status.error}
          </div>
        )}

        {!isActive && (
          <div className="flex items-end gap-2">
            <div className="flex-1 space-y-1.5">
              <Label className="text-xs">Max Rounds</Label>
              <Input
                type="number"
                min={1}
                max={100}
                value={maxRoundsInput}
                onChange={(e) => setMaxRoundsInput(Number(e.target.value))}
                className="h-9"
              />
            </div>
            <Button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              size="sm"
              className="h-9"
            >
              {startMutation.isPending ? (
                <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="mr-1 h-3.5 w-3.5" />
              )}
              Start
            </Button>
          </div>
        )}

        {isActive && status?.status !== 'stopping' && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => stopMutation.mutate()}
            disabled={stopMutation.isPending}
            className="w-full"
          >
            {stopMutation.isPending ? (
              <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Square className="mr-1 h-3.5 w-3.5" />
            )}
            Stop
          </Button>
        )}

        {status?.status === 'stopping' && (
          <p className="text-center text-xs text-muted-foreground">Stopping gracefully...</p>
        )}
      </CardContent>
    </Card>
  )
}

export default function RDAgentPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const qc = useQueryClient()

  const [maxRounds, setMaxRounds] = useState<Record<Market, number>>({
    us: 30,
    cn: 30,
    hk: 30,
  })

  const [factorMarketFilter, setFactorMarketFilter] = useState<Market | 'all'>('all')

  // Factors query
  const { data: factorsResp, isLoading: factorsLoading } = useQuery({
    queryKey: ['rdagent-factors', factorMarketFilter === 'all' ? undefined : factorMarketFilter],
    queryFn: () => rdagentApi.getFactors(factorMarketFilter === 'all' ? undefined : factorMarketFilter),
    select: (resp) => resp.data,
    staleTime: 30_000,
  })

  const factors = factorsResp?.factors ?? []

  // Toggle factor mutation
  const toggleMutation = useMutation({
    mutationFn: ({ id, isActive }: { id: number; isActive: boolean }) =>
      rdagentApi.toggleFactor(String(id), isActive),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['rdagent-factors'] })
    },
    onError: (err) => {
      toast({ title: 'Toggle failed', description: getErrorMessage(err), variant: 'destructive' })
    },
  })

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{t('nav.rdagent')}</h2>
        <p className="text-muted-foreground">Automated factor discovery and research</p>
      </div>

      {/* Per-market control cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {MARKETS.map((m) => (
          <MarketControlCard
            key={m}
            market={m}
            maxRoundsInput={maxRounds[m]}
            setMaxRoundsInput={(v) => setMaxRounds((prev) => ({ ...prev, [m]: v }))}
          />
        ))}
      </div>

      {/* Discovered Factors Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Zap className="h-5 w-5" />
                Discovered Factors
              </CardTitle>
              <CardDescription>{factors.length} factors total</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex gap-1 rounded-md border p-0.5">
                {(['all', ...MARKETS] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setFactorMarketFilter(m)}
                    className={cn(
                      'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                      factorMarketFilter === m
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                    )}
                  >
                    {m === 'all' ? 'All' : m.toUpperCase()}
                  </button>
                ))}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => qc.invalidateQueries({ queryKey: ['rdagent-factors'] })}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {factorsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : factors.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              <Brain className="mx-auto mb-2 h-8 w-8 opacity-40" />
              <p>No factors discovered yet. Start an RD-Agent run above.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2">Name</th>
                    <th className="pb-2">Expression</th>
                    <th className="pb-2">Market</th>
                    <th className="pb-2 text-right">IC</th>
                    <th className="pb-2 text-right">ICIR</th>
                    <th className="pb-2 text-center">Active</th>
                    <th className="pb-2">Discovered</th>
                  </tr>
                </thead>
                <tbody>
                  {factors.map((f) => (
                    <tr key={f.id} className="border-b last:border-0 hover:bg-muted/50">
                      <td className="py-2.5 font-medium">{f.name}</td>
                      <td className="max-w-[200px] truncate py-2.5 font-mono text-xs" title={f.expression}>
                        {f.expression}
                      </td>
                      <td className="py-2.5 uppercase">{f.market}</td>
                      <td className="py-2.5 text-right">
                        <IcBadge value={f.ic} />
                      </td>
                      <td className="py-2.5 text-right font-mono">
                        {f.icir.toFixed(4)}
                      </td>
                      <td className="py-2.5 text-center">
                        <button
                          onClick={() => toggleMutation.mutate({ id: f.id, isActive: !f.isActive })}
                          disabled={toggleMutation.isPending}
                          className="transition-colors hover:opacity-80"
                          title={f.isActive ? 'Click to deactivate' : 'Click to activate'}
                        >
                          {f.isActive ? (
                            <ToggleRight className="h-5 w-5 text-green-600 dark:text-green-400" />
                          ) : (
                            <ToggleLeft className="h-5 w-5 text-muted-foreground" />
                          )}
                        </button>
                      </td>
                      <td className="py-2.5 text-muted-foreground">
                        {formatDate(f.createdAt)}
                      </td>
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
