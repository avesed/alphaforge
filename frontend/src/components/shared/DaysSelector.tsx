import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const DEFAULT_OPTIONS = [7, 30, 90, 180] as const

type DaysOption = (typeof DEFAULT_OPTIONS)[number]

interface DaysSelectorProps {
  value: number
  onChange: (days: number) => void
  options?: readonly number[]
  className?: string
  size?: 'default' | 'sm'
}

export function DaysSelector({
  value,
  onChange,
  options = DEFAULT_OPTIONS,
  className,
  size = 'sm',
}: DaysSelectorProps) {
  const { t } = useTranslation()

  return (
    <div className={cn('inline-flex rounded-md shadow-sm', className)}>
      {options.map((days) => (
        <Button
          key={days}
          variant={value === days ? 'default' : 'outline'}
          size={size}
          onClick={() => onChange(days)}
          className={cn(
            'rounded-none first:rounded-l-md last:rounded-r-md',
            'border-r-0 last:border-r',
            value === days && 'z-10'
          )}
        >
          {t('common.nDays', { count: days })}
        </Button>
      ))}
    </div>
  )
}

export type { DaysOption }
