import {
  getStoredTheme,
  setStoredTheme,
  getSystemTheme,
  getResolvedTheme,
  applyTheme,
  subscribeToSystemThemeChanges,
} from './theme'

// ---------------------------------------------------------------------------
// getStoredTheme()
// ---------------------------------------------------------------------------
describe('getStoredTheme()', () => {
  it('returns "system" when nothing is stored', () => {
    expect(getStoredTheme()).toBe('system')
  })

  it('returns "light" when "light" is stored', () => {
    localStorage.setItem('alphaforge-theme', 'light')
    expect(getStoredTheme()).toBe('light')
  })

  it('returns "dark" when "dark" is stored', () => {
    localStorage.setItem('alphaforge-theme', 'dark')
    expect(getStoredTheme()).toBe('dark')
  })

  it('returns "system" for an invalid stored value', () => {
    localStorage.setItem('alphaforge-theme', 'sepia')
    expect(getStoredTheme()).toBe('system')
  })
})

// ---------------------------------------------------------------------------
// setStoredTheme()
// ---------------------------------------------------------------------------
describe('setStoredTheme()', () => {
  it('writes to localStorage', () => {
    setStoredTheme('dark')
    expect(localStorage.getItem('alphaforge-theme')).toBe('dark')
  })

  it('overwrites previous value', () => {
    setStoredTheme('light')
    setStoredTheme('dark')
    expect(localStorage.getItem('alphaforge-theme')).toBe('dark')
  })
})

// ---------------------------------------------------------------------------
// getSystemTheme()
// ---------------------------------------------------------------------------
describe('getSystemTheme()', () => {
  it('returns "light" when prefers-color-scheme: dark does not match', () => {
    // Default mock returns matches: false
    expect(getSystemTheme()).toBe('light')
  })

  it('returns "dark" when prefers-color-scheme: dark matches', () => {
    vi.mocked(window.matchMedia).mockReturnValueOnce({
      matches: true,
      media: '(prefers-color-scheme: dark)',
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })
    expect(getSystemTheme()).toBe('dark')
  })
})

// ---------------------------------------------------------------------------
// getResolvedTheme()
// ---------------------------------------------------------------------------
describe('getResolvedTheme()', () => {
  it('resolves "system" to the system theme (light by default)', () => {
    expect(getResolvedTheme('system')).toBe('light')
  })

  it('passes through "light" as-is', () => {
    expect(getResolvedTheme('light')).toBe('light')
  })

  it('passes through "dark" as-is', () => {
    expect(getResolvedTheme('dark')).toBe('dark')
  })
})

// ---------------------------------------------------------------------------
// applyTheme()
// ---------------------------------------------------------------------------
describe('applyTheme()', () => {
  it('adds "light" class and removes "dark" when theme is light', () => {
    applyTheme('light')
    expect(document.documentElement.classList.contains('light')).toBe(true)
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('adds "dark" class and removes "light" when theme is dark', () => {
    applyTheme('dark')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(document.documentElement.classList.contains('light')).toBe(false)
  })

  it('resolves "system" to the system preference before applying', () => {
    // Default mock: matches=false => system = light
    applyTheme('system')
    expect(document.documentElement.classList.contains('light')).toBe(true)
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// subscribeToSystemThemeChanges()
// ---------------------------------------------------------------------------
describe('subscribeToSystemThemeChanges()', () => {
  it('returns an unsubscribe function', () => {
    const cb = vi.fn()
    const unsub = subscribeToSystemThemeChanges(cb)
    expect(typeof unsub).toBe('function')
    unsub()
  })
})
