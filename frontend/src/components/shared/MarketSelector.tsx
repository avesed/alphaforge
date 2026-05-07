import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { Market } from '@/types'

const MARKETS: Market[] = ['us', 'cn', 'hk']

interface MarketSelectorProps {
  value: Market
  onChange: (market: Market) => void
  className?: string
  size?: 'default' | 'sm'
}

export function MarketSelector({
  value,
  onChange,
  className,
  size = 'sm',
}: MarketSelectorProps) {
  const { t } = useTranslation()

  return (
    <div className={cn('inline-flex rounded-md shadow-sm', className)}>
      {MARKETS.map((market) => (
        <Button
          key={market}
          variant={value === market ? 'default' : 'outline'}
          size={size}
          onClick={() => onChange(market)}
          className={cn(
            'rounded-none first:rounded-l-md last:rounded-r-md',
            'border-r-0 last:border-r',
            value === market && 'z-10'
          )}
        >
          {t(`market.${market}`)}
        </Button>
      ))}
    </div>
  )
}
