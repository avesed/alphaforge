import { ArrowUp, ArrowDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

type Direction = 'up' | 'down' | 'neutral'

const directionConfig: Record<
  Direction,
  { icon: typeof ArrowUp; bg: string; text: string }
> = {
  up: {
    icon: ArrowUp,
    bg: 'bg-green-100 dark:bg-green-900/30',
    text: 'text-green-700 dark:text-green-400',
  },
  down: {
    icon: ArrowDown,
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-700 dark:text-red-400',
  },
  neutral: {
    icon: Minus,
    bg: 'bg-gray-100 dark:bg-gray-800/50',
    text: 'text-gray-600 dark:text-gray-400',
  },
}

interface DirectionBadgeProps {
  direction: Direction
  probability?: number
  className?: string
  size?: 'sm' | 'default'
}

export function DirectionBadge({
  direction,
  probability,
  className,
  size = 'default',
}: DirectionBadgeProps) {
  const config = directionConfig[direction]
  const Icon = config.icon

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full font-medium',
        config.bg,
        config.text,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm',
        className
      )}
    >
      <Icon className={size === 'sm' ? 'h-3 w-3' : 'h-3.5 w-3.5'} />
      {probability !== undefined && (
        <span>{(probability * 100).toFixed(0)}%</span>
      )}
    </span>
  )
}

export type { Direction }
