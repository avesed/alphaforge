import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Settings,
  Save,
  Loader2,
  RefreshCw,
  Play,
  Crown,
  Clock,
  AlertTriangle,
  Lock,
  Link2,
  Newspaper,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { useToast } from '@/hooks/useToast'
import { getErrorMessage } from '@/api/client'
import { settingsApi } from '@/api/settings'
import type { SettingsMap } from '@/api/settings'
import { changePassword } from '@/api/auth'
import { schedulerApi } from '@/api/scheduler'
import { cn, formatDate } from '@/lib/utils'

// ── ML Settings section ──────────────────────

interface SettingFieldConfig {
  key: string
  label: string
  description: string
  type: 'boolean' | 'number'
  min?: number
  max?: number
  step?: number
}

const ML_SETTINGS: SettingFieldConfig[] = [
  {
    key: 'prediction_enabled',
    label: 'Prediction Enabled',
    description: 'Enable or disable the ML prediction pipeline',
    type: 'boolean',
  },
  {
    key: 'auto_retrain_enabled',
    label: 'Auto Retrain',
    description: 'Automatically retrain models when performance decays',
    type: 'boolean',
  },
  {
    key: 'ensemble_size',
    label: 'Ensemble Size',
    description: 'Number of models in the ensemble (1-10)',
    type: 'number',
    min: 1,
    max: 10,
    step: 1,
  },
  {
    key: 'walkforward_folds',
    label: 'Walk-Forward Folds',
    description: 'Number of validation folds (1-5)',
    type: 'number',
    min: 1,
    max: 5,
    step: 1,
  },
  {
    key: 'prediction_min_ic',
    label: 'Minimum IC',
    description: 'Quality gate IC threshold',
    type: 'number',
    min: 0,
    max: 0.1,
    step: 0.001,
  },
  {
    key: 'prediction_min_icir',
    label: 'Minimum ICIR',
    description: 'Quality gate ICIR threshold',
    type: 'number',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

function ToggleSwitch({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
        checked ? 'bg-primary' : 'bg-muted'
      )}
    >
      <span
        className={cn(
          'pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform',
          checked ? 'translate-x-5' : 'translate-x-0'
        )}
      />
    </button>
  )
}

function MLSettingsSection() {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [localSettings, setLocalSettings] = useState<SettingsMap>({})
  const [hasChanges, setHasChanges] = useState(false)

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.getAll().then((r) => r.data),
  })

  // Sync remote settings into local state
  useEffect(() => {
    if (settingsQuery.data != null) {
      setLocalSettings(settingsQuery.data)
      setHasChanges(false)
    }
  }, [settingsQuery.data])

  const saveMutation = useMutation({
    mutationFn: (settings: SettingsMap) => settingsApi.update(settings),
    onSuccess: () => {
      toast({ title: 'Settings saved' })
      setHasChanges(false)
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
    onError: (err) => {
      toast({
        title: 'Save failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const updateField = (key: string, value: string) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  if (settingsQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="flex items-center justify-between">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-8 w-24" />
            </div>
          ))}
        </CardContent>
      </Card>
    )
  }

  if (settingsQuery.isError) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center text-sm text-destructive">
            Failed to load settings.{' '}
            <button
              className="underline"
              onClick={() => void settingsQuery.refetch()}
            >
              Retry
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Settings className="h-4 w-4" />
          ML Configuration
        </CardTitle>
        <CardDescription>
          Model training and quality gate parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {ML_SETTINGS.map((field) => {
          const rawValue = localSettings[field.key] ?? ''

          if (field.type === 'boolean') {
            const checked = rawValue === 'true' || rawValue === '1'
            return (
              <div
                key={field.key}
                className="flex items-center justify-between gap-4"
              >
                <div>
                  <Label className="text-sm font-medium">{field.label}</Label>
                  <p className="text-xs text-muted-foreground">
                    {field.description}
                  </p>
                </div>
                <ToggleSwitch
                  checked={checked}
                  onChange={(v) => updateField(field.key, v ? 'true' : 'false')}
                />
              </div>
            )
          }

          return (
            <div
              key={field.key}
              className="flex items-center justify-between gap-4"
            >
              <div className="flex-1">
                <Label className="text-sm font-medium">{field.label}</Label>
                <p className="text-xs text-muted-foreground">
                  {field.description}
                </p>
              </div>
              <Input
                type="number"
                className="w-28 text-right font-mono"
                value={rawValue}
                onChange={(e) => updateField(field.key, e.target.value)}
                min={field.min}
                max={field.max}
                step={field.step}
              />
            </div>
          )
        })}

        <div className="flex items-center justify-between border-t pt-4">
          {hasChanges && (
            <span className="flex items-center gap-1 text-xs text-yellow-600 dark:text-yellow-400">
              <AlertTriangle className="h-3 w-3" />
              Unsaved changes
            </span>
          )}
          <div className="ml-auto flex gap-2">
            {hasChanges && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (settingsQuery.data != null) {
                    setLocalSettings(settingsQuery.data)
                    setHasChanges(false)
                  }
                }}
              >
                Discard
              </Button>
            )}
            <Button
              size="sm"
              disabled={!hasChanges || saveMutation.isPending}
              onClick={() => saveMutation.mutate(localSettings)}
            >
              {saveMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              Save Changes
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ── Scheduler section ──────────────────────

function SchedulerSection() {
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const schedulerQuery = useQuery({
    queryKey: ['scheduler'],
    queryFn: () => schedulerApi.getJobs().then((r) => r.data),
  })

  const triggerMutation = useMutation({
    mutationFn: (jobId: string) => schedulerApi.triggerJob(jobId),
    onSuccess: (_data, jobId) => {
      toast({ title: 'Job triggered', description: `Triggered job: ${jobId}` })
      void queryClient.invalidateQueries({ queryKey: ['scheduler'] })
    },
    onError: (err) => {
      toast({
        title: 'Trigger failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const relinquishMutation = useMutation({
    mutationFn: () => schedulerApi.relinquish(),
    onSuccess: () => {
      toast({ title: 'Leadership relinquished' })
      void queryClient.invalidateQueries({ queryKey: ['scheduler'] })
    },
    onError: (err) => {
      toast({
        title: 'Relinquish failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  if (schedulerQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </CardContent>
      </Card>
    )
  }

  if (schedulerQuery.isError) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center text-sm text-destructive">
            Failed to load scheduler info.{' '}
            <button
              className="underline"
              onClick={() => void schedulerQuery.refetch()}
            >
              Retry
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const scheduler = schedulerQuery.data
  if (scheduler == null) return null

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <Clock className="h-4 w-4" />
              Scheduler
            </CardTitle>
            <CardDescription>
              {scheduler.jobs.length} scheduled jobs
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {scheduler.isLeader ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2.5 py-1 text-xs font-medium text-green-800 dark:bg-green-900/50 dark:text-green-300">
                <Crown className="h-3 w-3" />
                Leader
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
                Follower
              </span>
            )}
            {scheduler.isLeader && (
              <Button
                variant="outline"
                size="sm"
                disabled={relinquishMutation.isPending}
                onClick={() => relinquishMutation.mutate()}
              >
                {relinquishMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Crown className="mr-2 h-4 w-4" />
                )}
                Relinquish
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => void schedulerQuery.refetch()}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {scheduler.jobs.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No scheduled jobs found.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="pb-2 pr-4 font-medium">Job Name</th>
                  <th className="pb-2 pr-4 font-medium">Next Run</th>
                  <th className="pb-2 pr-4 font-medium">Trigger</th>
                  <th className="pb-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {scheduler.jobs.map((job) => (
                  <tr
                    key={job.id}
                    className="border-b border-border/50 last:border-0"
                  >
                    <td className="py-2.5 pr-4 font-medium">{job.name}</td>
                    <td className="py-2.5 pr-4 font-mono text-xs text-muted-foreground">
                      {job.nextRunTime != null
                        ? formatDate(job.nextRunTime, {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        : 'Not scheduled'}
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-muted-foreground">
                      {job.trigger ?? '--'}
                    </td>
                    <td className="py-2.5">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        disabled={triggerMutation.isPending}
                        onClick={() => triggerMutation.mutate(job.id)}
                      >
                        {triggerMutation.isPending &&
                        triggerMutation.variables === job.id ? (
                          <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                        ) : (
                          <Play className="mr-1 h-3 w-3" />
                        )}
                        Trigger Now
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ── Password Change section ──────────────────────

function PasswordChangeSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const mismatch = confirmPassword.length > 0 && newPassword !== confirmPassword

  const passwordMutation = useMutation({
    mutationFn: () => changePassword({ currentPassword, newPassword }),
    onSuccess: () => {
      toast({ title: t('settings.passwordChanged') })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    },
    onError: (err) => {
      toast({
        title: t('settings.passwordChangeFailed'),
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const canSubmit =
    currentPassword.length > 0 &&
    newPassword.length >= 6 &&
    newPassword === confirmPassword &&
    !passwordMutation.isPending

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Lock className="h-4 w-4" />
          {t('settings.changePassword')}
        </CardTitle>
        <CardDescription>{t('settings.changePasswordDesc')}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.currentPassword')}
          </Label>
          <Input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.newPassword')}
          </Label>
          <Input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.confirmPassword')}
          </Label>
          <Input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
          {mismatch && (
            <p className="text-xs text-destructive">
              {t('settings.passwordMismatch')}
            </p>
          )}
        </div>
        <div className="flex justify-end border-t pt-4">
          <Button
            size="sm"
            disabled={!canSubmit}
            onClick={() => passwordMutation.mutate()}
          >
            {passwordMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            {t('settings.changePassword')}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// ── StockPulse Settings section ──────────────────────

function StockPulseSettingsSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [hasChanges, setHasChanges] = useState(false)

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.getAll().then((r) => r.data),
  })

  useEffect(() => {
    if (settingsQuery.data != null) {
      setUrl(settingsQuery.data['stockpulse_url'] ?? '')
      setApiKey(settingsQuery.data['stockpulse_api_key'] ?? '')
      setHasChanges(false)
    }
  }, [settingsQuery.data])

  const existingKey = settingsQuery.data?.['stockpulse_api_key'] ?? ''
  const keyHint =
    existingKey.length >= 4 ? existingKey.slice(-4) : ''

  const testMutation = useMutation({
    mutationFn: () => settingsApi.testStockPulse().then((r) => r.data),
    onSuccess: (data) => {
      if (data.connected) {
        toast({ title: t('settings.stockpulseConnected') })
      } else {
        toast({
          title: t('settings.stockpulseDisconnected'),
          description: data.error,
          variant: 'destructive',
        })
      }
    },
    onError: (err) => {
      toast({
        title: t('settings.stockpulseDisconnected'),
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const saveMutation = useMutation({
    mutationFn: () =>
      settingsApi.update({
        stockpulse_url: url,
        stockpulse_api_key: apiKey,
      }),
    onSuccess: () => {
      toast({ title: t('settings.saved') })
      setHasChanges(false)
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
    onError: (err) => {
      toast({
        title: 'Save failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  if (settingsQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-56" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Link2 className="h-4 w-4" />
          {t('settings.stockpulseConnection')}
        </CardTitle>
        <CardDescription>
          {t('settings.stockpulseConnectionDesc')}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.stockpulseUrl')}
          </Label>
          <Input
            type="text"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              setHasChanges(true)
            }}
            placeholder="https://..."
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.stockpulseApiKey')}
          </Label>
          <Input
            type="password"
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value)
              setHasChanges(true)
            }}
          />
          {keyHint && (
            <p className="text-xs text-muted-foreground">
              {t('settings.stockpulseApiKeyHint', { suffix: keyHint })}
            </p>
          )}
        </div>
        <div className="flex items-center justify-between border-t pt-4">
          <Button
            variant="outline"
            size="sm"
            disabled={testMutation.isPending}
            onClick={() => testMutation.mutate()}
          >
            {testMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            {t('settings.testConnection')}
          </Button>
          <Button
            size="sm"
            disabled={!hasChanges || saveMutation.isPending}
            onClick={() => saveMutation.mutate()}
          >
            {saveMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            {t('common.save')}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// ── NewsForge Connection section ──────────────

function NewsForgeSettingsSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [hasChanges, setHasChanges] = useState(false)

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.getAll().then((r) => r.data),
  })

  useEffect(() => {
    if (settingsQuery.data != null) {
      setUrl(settingsQuery.data['newsforge_url'] ?? '')
      setApiKey(settingsQuery.data['newsforge_api_key'] ?? '')
      setHasChanges(false)
    }
  }, [settingsQuery.data])

  const existingKey = settingsQuery.data?.['newsforge_api_key'] ?? ''
  const keyHint =
    existingKey.length >= 4 ? existingKey.slice(-4) : ''

  const testMutation = useMutation({
    mutationFn: () => settingsApi.testNewsForge().then((r) => r.data),
    onSuccess: (data) => {
      if (data.connected) {
        toast({ title: t('settings.newsforgeConnected') })
      } else {
        toast({
          title: t('settings.newsforgeDisconnected'),
          description: data.error,
          variant: 'destructive',
        })
      }
    },
    onError: (err) => {
      toast({
        title: t('settings.newsforgeDisconnected'),
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  const saveMutation = useMutation({
    mutationFn: () =>
      settingsApi.update({
        newsforge_url: url,
        newsforge_api_key: apiKey,
      }),
    onSuccess: () => {
      toast({ title: t('settings.saved') })
      setHasChanges(false)
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
    onError: (err) => {
      toast({
        title: 'Save failed',
        description: getErrorMessage(err),
        variant: 'destructive',
      })
    },
  })

  if (settingsQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-56" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Newspaper className="h-4 w-4" />
          {t('settings.newsforgeConnection')}
        </CardTitle>
        <CardDescription>
          {t('settings.newsforgeConnectionDesc')}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.newsforgeUrl')}
          </Label>
          <Input
            type="text"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              setHasChanges(true)
            }}
            placeholder="https://..."
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {t('settings.newsforgeApiKey')}
          </Label>
          <Input
            type="password"
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value)
              setHasChanges(true)
            }}
          />
          {keyHint && (
            <p className="text-xs text-muted-foreground">
              {t('settings.newsforgeApiKeyHint', { suffix: keyHint })}
            </p>
          )}
        </div>
        <div className="flex items-center justify-between border-t pt-4">
          <Button
            variant="outline"
            size="sm"
            disabled={testMutation.isPending}
            onClick={() => testMutation.mutate()}
          >
            {testMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            {t('settings.testConnection')}
          </Button>
          <Button
            size="sm"
            disabled={!hasChanges || saveMutation.isPending}
            onClick={() => saveMutation.mutate()}
          >
            {saveMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            {t('common.save')}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// ── Main Page ──────────────────────────────

export default function SettingsPage() {
  const { t } = useTranslation()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">
          <Settings className="mr-2 inline-block h-6 w-6" />
          {t('nav.settings')}
        </h2>
        <p className="text-muted-foreground">
          System configuration and scheduler management
        </p>
      </div>

      <MLSettingsSection />
      <StockPulseSettingsSection />
      <NewsForgeSettingsSection />
      <PasswordChangeSection />
      <SchedulerSection />
    </div>
  )
}
