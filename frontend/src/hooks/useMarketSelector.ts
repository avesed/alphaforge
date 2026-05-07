import { useState } from 'react'
import type { Market } from '@/types'

export function useMarketSelector(defaultMarket: Market = 'us') {
  const [market, setMarket] = useState<Market>(defaultMarket)
  return { market, setMarket } as const
}
