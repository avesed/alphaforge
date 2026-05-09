import { cn, formatDate, formatNumber, getErrorMessage } from './utils'

// ---------------------------------------------------------------------------
// cn() — class merging with Tailwind conflict resolution
// ---------------------------------------------------------------------------
describe('cn()', () => {
  it('merges multiple class strings', () => {
    expect(cn('foo', 'bar')).toBe('foo bar')
  })

  it('resolves Tailwind conflicts by keeping the last value', () => {
    expect(cn('p-4', 'p-2')).toBe('p-2')
  })

  it('merges conditional classes via clsx syntax', () => {
    expect(cn('base', false && 'hidden', 'extra')).toBe('base extra')
  })

  it('handles undefined and null inputs gracefully', () => {
    expect(cn('a', undefined, null, 'b')).toBe('a b')
  })

  it('handles empty arguments', () => {
    expect(cn()).toBe('')
  })

  it('resolves conflicting Tailwind text colors', () => {
    expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500')
  })
})

// ---------------------------------------------------------------------------
// formatDate()
// ---------------------------------------------------------------------------
describe('formatDate()', () => {
  it('formats a Date object with default options', () => {
    const result = formatDate(new Date('2024-03-15T00:00:00Z'))
    // Default locale is en-US, format is "month short, day, year"
    expect(result).toContain('2024')
    expect(result).toContain('15')
  })

  it('formats a string date with default options', () => {
    const result = formatDate('2024-01-01T00:00:00Z')
    expect(result).toContain('2024')
    expect(result).toContain('Jan')
  })

  it('respects custom locale (zh-CN)', () => {
    const result = formatDate('2024-06-15T00:00:00Z', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }, 'zh-CN')
    // Chinese locale uses different formatting
    expect(result).toContain('2024')
  })

  it('respects custom format options', () => {
    const result = formatDate('2024-03-15T00:00:00Z', {
      year: 'numeric',
      month: 'long',
      day: '2-digit',
    })
    expect(result).toContain('March')
  })
})

// ---------------------------------------------------------------------------
// formatNumber()
// ---------------------------------------------------------------------------
describe('formatNumber()', () => {
  it('formats an integer with no decimals by default', () => {
    expect(formatNumber(1234)).toBe('1,234')
  })

  it('formats with specified decimal places', () => {
    expect(formatNumber(1234.5678, 2)).toBe('1,234.57')
  })

  it('pads with trailing zeros when decimals requested', () => {
    expect(formatNumber(10, 3)).toBe('10.000')
  })

  it('respects a different locale (de-DE)', () => {
    const result = formatNumber(1234.56, 2, 'de-DE')
    // German uses . for thousands and , for decimals
    expect(result).toContain('1.234')
  })

  it('handles zero', () => {
    expect(formatNumber(0)).toBe('0')
  })
})

// ---------------------------------------------------------------------------
// getErrorMessage()
// ---------------------------------------------------------------------------
describe('getErrorMessage()', () => {
  it('extracts message from Error instance', () => {
    expect(getErrorMessage(new Error('Something broke'))).toBe('Something broke')
  })

  it('returns fallback for a plain string', () => {
    expect(getErrorMessage('random string')).toBe('An unexpected error occurred')
  })

  it('returns fallback for null', () => {
    expect(getErrorMessage(null)).toBe('An unexpected error occurred')
  })

  it('returns fallback for undefined', () => {
    expect(getErrorMessage(undefined)).toBe('An unexpected error occurred')
  })

  it('returns fallback for a number', () => {
    expect(getErrorMessage(42)).toBe('An unexpected error occurred')
  })
})
