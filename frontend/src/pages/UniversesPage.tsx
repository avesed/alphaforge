import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Globe2, Plus, X, Database,
  CheckCircle2,
} from 'lucide-react'

import type { Market, PredictionUniverse } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { cn, formatDate } from '@/lib/utils'

const MARKETS: Market[] = ['us', 'cn', 'hk']

type UniverseType = 'index' | 'custom'

// Example universe data -- in production these would come from a StockPulse API
const EXAMPLE_UNIVERSES: PredictionUniverse[] = [
  {
    id: 1,
    name: 'CSI 300',
    market: 'cn',
    universeType: 'index',
    symbols: Array.from({ length: 300 }, (_, i) => `6${String(i).padStart(5, '0')}.SH`),
    createdAt: '2026-01-15T08:00:00Z',
  },
  {
    id: 2,
    name: 'S&P 500',
    market: 'us',
    universeType: 'index',
    symbols: Array.from({ length: 500 }, (_, i) => `STOCK${i}`),
    createdAt: '2026-01-15T08:00:00Z',
  },
  {
    id: 3,
    name: 'HSI',
    market: 'hk',
    universeType: 'index',
    symbols: Array.from({ length: 80 }, (_, i) => `0${String(i).padStart(4, '0')}.HK`),
    createdAt: '2026-01-15T08:00:00Z',
  },
]

function MarketBadge({ market }: { market: Market }) {
  const colors: Record<Market, string> = {
    us: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    cn: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    hk: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
  }
  return (
    <span className={cn('rounded-full px-2 py-0.5 text-xs font-medium uppercase', colors[market])}>
      {market}
    </span>
  )
}

function TypeBadge({ type }: { type: string }) {
  return (
    <span className={cn(
      'rounded-full px-2 py-0.5 text-xs font-medium',
      type === 'index'
        ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400'
        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
    )}>
      {type}
    </span>
  )
}

export default function UniversesPage() {
  const { t } = useTranslation()
  const [showCreateForm, setShowCreateForm] = useState(false)

  // Create form state
  const [newName, setNewName] = useState('')
  const [newMarket, setNewMarket] = useState<Market>('us')
  const [newType, setNewType] = useState<UniverseType>('index')
  const [newIndexCode, setNewIndexCode] = useState('')
  const [newSymbols, setNewSymbols] = useState('')

  const universes = EXAMPLE_UNIVERSES

  const handleCreate = () => {
    // This would call a universes API when available
    // For now, just show the form to demonstrate the UI
    setShowCreateForm(false)
    setNewName('')
    setNewIndexCode('')
    setNewSymbols('')
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">{t('nav.universes')}</h2>
          <p className="text-muted-foreground">Manage stock universes for prediction models</p>
        </div>
        <Button
          onClick={() => setShowCreateForm(!showCreateForm)}
          variant={showCreateForm ? 'outline' : 'default'}
        >
          {showCreateForm ? (
            <>
              <X className="mr-2 h-4 w-4" />
              Cancel
            </>
          ) : (
            <>
              <Plus className="mr-2 h-4 w-4" />
              Create Universe
            </>
          )}
        </Button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">New Universe</CardTitle>
            <CardDescription>Define a stock universe for training and prediction</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="univName">Name</Label>
                <Input
                  id="univName"
                  placeholder="e.g., CSI 500"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Market</Label>
                <div className="flex gap-1 rounded-md border p-0.5">
                  {MARKETS.map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setNewMarket(m)}
                      className={cn(
                        'flex-1 rounded px-3 py-1.5 text-sm font-medium transition-colors',
                        newMarket === m
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                      )}
                    >
                      {m.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label>Type</Label>
                <div className="flex gap-1 rounded-md border p-0.5">
                  {(['index', 'custom'] as const).map((type) => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setNewType(type)}
                      className={cn(
                        'flex-1 rounded px-3 py-1.5 text-sm font-medium capitalize transition-colors',
                        newType === type
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                      )}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {newType === 'index' ? (
              <div className="space-y-2">
                <Label htmlFor="indexCode">Index Code</Label>
                <Input
                  id="indexCode"
                  placeholder="e.g., 000300.SH, SPX, HSI"
                  value={newIndexCode}
                  onChange={(e) => setNewIndexCode(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Symbols will be synced automatically from the index constituents.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <Label htmlFor="customSymbols">Symbols</Label>
                <textarea
                  id="customSymbols"
                  className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  placeholder="AAPL, MSFT, GOOGL, AMZN, META"
                  value={newSymbols}
                  onChange={(e) => setNewSymbols(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Enter symbols separated by commas. One symbol per line also works.
                </p>
              </div>
            )}

            <div className="flex gap-2">
              <Button onClick={handleCreate} disabled={!newName.trim()}>
                Create Universe
              </Button>
              <Button variant="outline" onClick={() => setShowCreateForm(false)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Info banner */}
      <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/20">
        <CardContent className="flex items-start gap-3 py-4">
          <Database className="mt-0.5 h-5 w-5 text-blue-500" />
          <div className="text-sm">
            <p className="font-medium text-foreground">Universes managed via StockPulse</p>
            <p className="text-muted-foreground">
              Prediction universes are stored in the StockPulse data platform. Default universes
              (CSI 300, S&P 500, HSI) are seeded automatically. Custom universes can be
              created and will be synced during prediction runs.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Universes Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Globe2 className="h-5 w-5" />
            Universes
          </CardTitle>
          <CardDescription>{universes.length} universes configured</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="pb-2">Name</th>
                  <th className="pb-2">Market</th>
                  <th className="pb-2">Type</th>
                  <th className="pb-2 text-right">Symbols</th>
                  <th className="pb-2 text-center">Default</th>
                  <th className="pb-2 text-center">Active</th>
                  <th className="pb-2">Created</th>
                </tr>
              </thead>
              <tbody>
                {universes.map((u) => (
                  <tr key={u.id} className="border-b last:border-0 hover:bg-muted/50">
                    <td className="py-3 font-medium">{u.name}</td>
                    <td className="py-3"><MarketBadge market={u.market} /></td>
                    <td className="py-3"><TypeBadge type={u.universeType} /></td>
                    <td className="py-3 text-right font-mono">{u.symbols.length}</td>
                    <td className="py-3 text-center">
                      <CheckCircle2 className="inline h-4 w-4 text-green-500" />
                    </td>
                    <td className="py-3 text-center">
                      <CheckCircle2 className="inline h-4 w-4 text-green-500" />
                    </td>
                    <td className="py-3 text-muted-foreground">{formatDate(u.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
