import apiClient from '@/api/client'
import type { User, TokenResponse } from '@/types'

export async function login(data: {
  email: string
  password: string
}): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>('/admin/auth/login', data)
  return response.data
}

export async function register(data: {
  email: string
  password: string
  displayName: string
}): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>('/admin/auth/register', data)
  return response.data
}

export async function logout(refreshToken: string | null): Promise<void> {
  await apiClient.post('/admin/auth/logout', { refreshToken })
}

export async function getMe(): Promise<User> {
  const response = await apiClient.get<User>('/admin/auth/me')
  return response.data
}

export async function changePassword(data: {
  currentPassword: string
  newPassword: string
}): Promise<void> {
  await apiClient.post('/admin/auth/change-password', data)
}
