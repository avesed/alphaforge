import { create } from 'zustand'
import type { User } from '@/types'
import * as authApi from '@/api/auth'
import {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
} from '@/api/client'

interface AuthState {
  user: User | null
  token: string | null
  isLoading: boolean
}

interface AuthActions {
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  initAuth: () => Promise<void>
  isAdmin: () => boolean
  isAuthenticated: () => boolean
}

export const useAuthStore = create<AuthState & AuthActions>((set, get) => ({
  user: null,
  token: getAccessToken(),
  isLoading: true,

  login: async (email, password) => {
    const response = await authApi.login({ email, password })
    setTokens(response.accessToken, response.refreshToken)
    set({ token: response.accessToken })
    const user = await authApi.getMe()
    set({ user })
  },

  logout: async () => {
    const refreshToken = getRefreshToken()
    try {
      await authApi.logout(refreshToken)
    } catch {
      // Ignore logout errors
    }
    clearTokens()
    set({ user: null, token: null })
  },

  initAuth: async () => {
    const token = getAccessToken()
    if (!token) {
      set({ isLoading: false })
      return
    }
    try {
      const user = await authApi.getMe()
      set({ user, token, isLoading: false })
    } catch {
      clearTokens()
      set({ user: null, token: null, isLoading: false })
    }
  },

  isAdmin: () => get().user?.role === 'admin',
  isAuthenticated: () => get().user !== null,
}))
