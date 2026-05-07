import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

type Trend = 'up' | 'down' | 'neutral'

const trendConfig: Record<Trend, { icon: typeof TrendingUp; color: string }> = {
  up: { icon: TrendingUp, color: 'text-green-600 dark:text-green-400' },
  down: { icon: TrendingDown, color: 'text-red-600 dark:text-red-400' },
  neutral: { icon: Minus, color: 'text-muted-foreground' },
}

interface MetricCardProps {
  label: string
  value: string | number
  trend?: Trend
  subtitle?: string
  className?: string
}

export function MetricCard({
  label,
  value,
  trend,
  subtitle,
  className,
}: MetricCardProps) {
  const TrendIcon = trend ? trendConfig[trend].icon : null

  return (
    <Card className={cn('', className)}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold tracking-tight">{value}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {TrendIcon && trend && (
            <TrendIcon
              className={cn('h-5 w-5', trendConfig[trend].color)}
            />
          )}
        </div>
      </CardContent>
    </Card>
  )
}
