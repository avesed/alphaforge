import { renderHook, act } from '@/test/test-utils'
import { useMarketSelector } from './useMarketSelector'

describe('useMarketSelector', () => {
  it('defaults to "us" when no argument is provided', () => {
    const { result } = renderHook(() => useMarketSelector())
    expect(result.current.market).toBe('us')
  })

  it('accepts a custom default market', () => {
    const { result } = renderHook(() => useMarketSelector('cn'))
    expect(result.current.market).toBe('cn')
  })

  it('setMarket updates the market value', () => {
    const { result } = renderHook(() => useMarketSelector())
    expect(result.current.market).toBe('us')

    act(() => {
      result.current.setMarket('hk')
    })

    expect(result.current.market).toBe('hk')
  })

  it('returns stable setMarket reference across renders', () => {
    const { result, rerender } = renderHook(() => useMarketSelector())

    const firstSetMarket = result.current.setMarket
    rerender()
    const secondSetMarket = result.current.setMarket

    expect(firstSetMarket).toBe(secondSetMarket)
  })
})
