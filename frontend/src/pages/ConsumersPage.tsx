import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Users,
  Plus,
  Copy,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Loader2,
  Key,
  Shield,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { getErrorMessage } from '@/api/client'
import { consumersApi } from '@/api/consumers'
import { formatDate } from '@/lib/utils'
import type { ApiConsumer } from '@/types'

function StatusBadge({ active }: { active: boolean }) {
  if (active) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900/50 dark:text-green-300">
        <CheckCircle2 className="h-3 w-3" />
        Active
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/50 dark:text-red-300">
      <XCircle className="h-3 w-3" />
      Inactive
    </span>
  )
}

function CreateConsumerForm({
  onClose,
}: {
  onClose: () => void
}) {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [rawKey, setRawKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const createMutation = useMutation({
    mutationFn: () =>
      consumersApi.create(name, description || undefined).then((r) => r.data),
    onSuccess: (data) => {
      setRawKey(data.rawApiKey)
      toast({ title: 'Consumer created', description: `API key generated for ${data.name}` })
      void queryClient.invalidateQueries({ queryKey: ['consumers'] })
    },
    onError: (err) => {
      toast({
        title: 'Create failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const handleCopy = async () => {
    if (rawKey == null) return
    try {
      await navigator.clipboard.writeText(rawKey)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast({
        title: 'Copy failed',
        description: 'Could not copy to clipboard',
        variant: 'destructive',
      })
    }
  }

  if (rawKey != null) {
    return (
      <Card className="border-yellow-500/50 dark:border-yellow-500/30">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4" />
            API Key Created
          </CardTitle>
          <CardDescription>
            This key will only be shown once. Copy it now and store it securely.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2">
            <code className="flex-1 rounded-md bg-muted p-3 text-xs font-mono break-all">
              {rawKey}
            </code>
            <Button variant="outline" size="icon" onClick={() => void handleCopy()}>
              {copied ? (
                <CheckCircle2 className="h-4 w-4 text-green-600" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          </div>
          <Button variant="outline" className="w-full" onClick={onClose}>
            Done
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Plus className="h-4 w-4" />
          Create API Consumer
        </CardTitle>
        <CardDescription>
          Generate a new API key for service integration
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            createMutation.mutate()
          }}
          className="space-y-4"
        >
          <div className="space-y-2">
            <Label htmlFor="consumer-name">Name</Label>
            <Input
              id="consumer-name"
              placeholder="e.g., WebStock Production"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="consumer-desc">Description (optional)</Label>
            <textarea
              id="consumer-desc"
              placeholder="What is this consumer used for?"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              rows={3}
            />
          </div>
          <div className="flex gap-2">
            <Button
              type="submit"
              disabled={createMutation.isPending || name.trim() === ''}
            >
              {createMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Key className="mr-2 h-4 w-4" />
              )}
              Generate Key
            </Button>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}

function ConsumerRow({ consumer }: { consumer: ApiConsumer }) {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [confirming, setConfirming] = useState(false)

  const deactivateMutation = useMutation({
    mutationFn: () => consumersApi.deactivate(consumer.id),
    onSuccess: () => {
      toast({ title: 'Consumer deactivated' })
      void queryClient.invalidateQueries({ queryKey: ['consumers'] })
      setConfirming(false)
    },
    onError: (err) => {
      toast({
        title: 'Deactivation failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
      setConfirming(false)
    },
  })

  return (
    <tr className="border-b border-border/50 last:border-0">
      <td className="p-4 font-medium">{consumer.name}</td>
      <td className="p-4 font-mono text-xs text-muted-foreground">
        {consumer.apiKeyPrefix}...
      </td>
      <td className="p-4 text-right">{consumer.rateLimit}/min</td>
      <td className="p-4 font-mono text-xs text-muted-foreground">
        {consumer.lastUsedAt != null
          ? formatDate(consumer.lastUsedAt)
          : 'Never'}
      </td>
      <td className="p-4">
        <StatusBadge active={consumer.isActive} />
      </td>
      <td className="p-4">
        {consumer.isActive && (
          <>
            {confirming ? (
              <div className="flex gap-1">
                <Button
                  variant="destructive"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={deactivateMutation.isPending}
                  onClick={() => deactivateMutation.mutate()}
                >
                  {deactivateMutation.isPending ? (
                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  ) : null}
                  Confirm
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => setConfirming(false)}
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-destructive hover:text-destructive"
                onClick={() => setConfirming(true)}
              >
                <XCircle className="mr-1 h-3 w-3" />
                Deactivate
              </Button>
            )}
          </>
        )}
      </td>
    </tr>
  )
}

export default function ConsumersPage() {
  const { t } = useTranslation()
  const [showCreate, setShowCreate] = useState(false)

  const consumersQuery = useQuery({
    queryKey: ['consumers'],
    queryFn: () => consumersApi.list().then((r) => r.data),
  })

  const consumers: ApiConsumer[] = consumersQuery.data?.consumers ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            <Users className="mr-2 inline-block h-6 w-6" />
            {t('nav.consumers')}
          </h2>
          <p className="text-muted-foreground">
            Manage API keys for external service integration
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void consumersQuery.refetch()}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            {t('common.refresh')}
          </Button>
          {!showCreate && (
            <Button size="sm" onClick={() => setShowCreate(true)}>
              <Plus className="mr-2 h-4 w-4" />
              New Consumer
            </Button>
          )}
        </div>
      </div>

      {showCreate && (
        <CreateConsumerForm onClose={() => setShowCreate(false)} />
      )}

      {consumersQuery.isLoading && (
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3">
              {Array.from({ length: 4 }, (_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {consumersQuery.isError && (
        <Card>
          <CardContent className="p-6">
            <div className="text-center text-sm text-destructive">
              Failed to load consumers.{' '}
              <button
                className="underline"
                onClick={() => void consumersQuery.refetch()}
              >
                Retry
              </button>
            </div>
          </CardContent>
        </Card>
      )}

      {consumersQuery.isSuccess && consumers.length === 0 && !showCreate && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-12">
            <Shield className="h-12 w-12 text-muted-foreground/50" />
            <div className="text-center">
              <p className="font-medium">No API consumers yet</p>
              <p className="text-sm text-muted-foreground">
                Create a consumer to generate API keys for WebStock or other
                services.
              </p>
            </div>
            <Button onClick={() => setShowCreate(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Create First Consumer
            </Button>
          </CardContent>
        </Card>
      )}

      {consumers.length > 0 && (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="p-4 font-medium">Name</th>
                    <th className="p-4 font-medium">Key Prefix</th>
                    <th className="p-4 font-medium text-right">Rate Limit</th>
                    <th className="p-4 font-medium">Last Used</th>
                    <th className="p-4 font-medium">Status</th>
                    <th className="p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {consumers.map((c) => (
                    <ConsumerRow key={c.id} consumer={c} />
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
