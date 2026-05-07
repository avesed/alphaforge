import { create } from 'zustand'
import type { Theme } from '@/lib/theme'
import {
  getStoredTheme,
  setStoredTheme,
  applyTheme,
  subscribeToSystemThemeChanges,
  getResolvedTheme,
} from '@/lib/theme'

interface ThemeState {
  theme: Theme
  resolvedTheme: 'light' | 'dark'
}

interface ThemeActions {
  setTheme: (theme: Theme) => void
  initTheme: () => void
}

type ThemeStore = ThemeState & ThemeActions

export const useThemeStore = create<ThemeStore>((set, get) => ({
  theme: 'system',
  resolvedTheme: 'light',

  setTheme: (theme: Theme) => {
    setStoredTheme(theme)
    applyTheme(theme)
    set({
      theme,
      resolvedTheme: getResolvedTheme(theme),
    })
  },

  initTheme: () => {
    const storedTheme = getStoredTheme()
    applyTheme(storedTheme)

    set({
      theme: storedTheme,
      resolvedTheme: getResolvedTheme(storedTheme),
    })

    subscribeToSystemThemeChanges((systemTheme) => {
      const currentTheme = get().theme
      if (currentTheme === 'system') {
        applyTheme('system')
        set({ resolvedTheme: systemTheme })
      }
    })
  },
}))
