import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios'
import type { ApiError } from '@/types'

type RetryableConfig = InternalAxiosRequestConfig & { _retry?: boolean }

const TOKEN_KEY = 'alphaforge-token'
const REFRESH_KEY = 'alphaforge-refresh'

export function getAccessToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_KEY)
}

export function setTokens(access: string, refresh: string): void {
  localStorage.setItem(TOKEN_KEY, access)
  localStorage.setItem(REFRESH_KEY, refresh)
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

let refreshPromise: Promise<string> | null = null

async function refreshAccessToken(): Promise<string> {
  if (refreshPromise) return refreshPromise

  refreshPromise = (async () => {
    try {
      const refreshToken = getRefreshToken()
      if (!refreshToken) throw new Error('No refresh token')

      const response = await axios.post<{ accessToken: string; refreshToken: string }>(
        '/api/v1/admin/auth/refresh',
        { refreshToken }
      )
      const { accessToken, refreshToken: newRefresh } = response.data
      setTokens(accessToken, newRefresh)
      return accessToken
    } finally {
      refreshPromise = null
    }
  })()

  return refreshPromise
}

const apiClient = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
})

apiClient.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as RetryableConfig | undefined
    const status = error.response?.status

    if (status !== 401 || !original || original._retry) {
      return Promise.reject(error)
    }

    if (original.url?.endsWith('/auth/refresh')) {
      clearTokens()
      window.location.href = '/login'
      return Promise.reject(error)
    }

    original._retry = true
    try {
      const newToken = await refreshAccessToken()
      original.headers = original.headers ?? {}
      original.headers.Authorization = `Bearer ${newToken}`
      return apiClient(original)
    } catch {
      clearTokens()
      window.location.href = '/login'
      return Promise.reject(error)
    }
  }
)

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as ApiError | undefined
    if (data?.detail) {
      return data.detail
    }
    if (error.message) {
      return error.message
    }
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unexpected error occurred'
}

export default apiClient
