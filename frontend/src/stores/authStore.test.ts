import { http, HttpResponse } from 'msw'
import { server } from '@/test/mocks/server'
import { useAuthStore } from './authStore'
import type { User, TokenResponse } from '@/types'

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------
const mockUser: User = {
  id: 1,
  email: 'admin@alphaforge.dev',
  displayName: 'Admin',
  role: 'admin',
  locale: 'en',
}

const mockRegularUser: User = {
  id: 2,
  email: 'user@alphaforge.dev',
  displayName: 'Regular',
  role: 'user',
  locale: 'zh',
}

const mockTokenResponse: TokenResponse = {
  accessToken: 'test-access-token',
  refreshToken: 'test-refresh-token',
  tokenType: 'bearer',
}

// ---------------------------------------------------------------------------
// Reset store between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  useAuthStore.setState({ user: null, token: null, isLoading: true })
})

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------
describe('Initial state', () => {
  it('user is null by default', () => {
    expect(useAuthStore.getState().user).toBeNull()
  })

  it('isLoading is true by default', () => {
    expect(useAuthStore.getState().isLoading).toBe(true)
  })

  it('token reads from localStorage on store creation', () => {
    // When store was created (module load), localStorage was empty, so token is null
    expect(useAuthStore.getState().token).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// isAdmin()
// ---------------------------------------------------------------------------
describe('isAdmin()', () => {
  it('returns true when user.role is admin', () => {
    useAuthStore.setState({ user: mockUser })
    expect(useAuthStore.getState().isAdmin()).toBe(true)
  })

  it('returns false when user.role is user', () => {
    useAuthStore.setState({ user: mockRegularUser })
    expect(useAuthStore.getState().isAdmin()).toBe(false)
  })

  it('returns false when user is null', () => {
    useAuthStore.setState({ user: null })
    expect(useAuthStore.getState().isAdmin()).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// isAuthenticated()
// ---------------------------------------------------------------------------
describe('isAuthenticated()', () => {
  it('returns true when user is set', () => {
    useAuthStore.setState({ user: mockUser })
    expect(useAuthStore.getState().isAuthenticated()).toBe(true)
  })

  it('returns false when user is null', () => {
    useAuthStore.setState({ user: null })
    expect(useAuthStore.getState().isAuthenticated()).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// login()
// ---------------------------------------------------------------------------
describe('login()', () => {
  it('calls login API, stores tokens, fetches user profile, and updates state', async () => {
    server.use(
      http.post('/api/v1/admin/auth/login', () => {
        return HttpResponse.json(mockTokenResponse)
      }),
      http.get('/api/v1/admin/auth/me', () => {
        return HttpResponse.json(mockUser)
      })
    )

    await useAuthStore.getState().login('admin@alphaforge.dev', 'Admin123')

    const state = useAuthStore.getState()
    expect(state.user).toEqual(mockUser)
    expect(state.token).toBe('test-access-token')
    expect(localStorage.getItem('alphaforge-token')).toBe('test-access-token')
    expect(localStorage.getItem('alphaforge-refresh')).toBe('test-refresh-token')
  })

  it('throws on failed login and does not set tokens', async () => {
    server.use(
      http.post('/api/v1/admin/auth/login', () => {
        return HttpResponse.json(
          { detail: 'Invalid credentials' },
          { status: 401 }
        )
      })
    )

    await expect(
      useAuthStore.getState().login('bad@email.com', 'wrong')
    ).rejects.toThrow()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(localStorage.getItem('alphaforge-token')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// logout()
// ---------------------------------------------------------------------------
describe('logout()', () => {
  it('calls logout API, clears tokens, and nulls state', async () => {
    // Pre-fill state as if logged in
    localStorage.setItem('alphaforge-token', 'access')
    localStorage.setItem('alphaforge-refresh', 'refresh')
    useAuthStore.setState({ user: mockUser, token: 'access', isLoading: false })

    server.use(
      http.post('/api/v1/admin/auth/logout', () => {
        return HttpResponse.json({ ok: true })
      })
    )

    await useAuthStore.getState().logout()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(localStorage.getItem('alphaforge-token')).toBeNull()
    expect(localStorage.getItem('alphaforge-refresh')).toBeNull()
  })

  it('ignores logout API errors gracefully', async () => {
    localStorage.setItem('alphaforge-token', 'access')
    localStorage.setItem('alphaforge-refresh', 'refresh')
    useAuthStore.setState({ user: mockUser, token: 'access', isLoading: false })

    server.use(
      http.post('/api/v1/admin/auth/logout', () => {
        return new HttpResponse(null, { status: 500 })
      })
    )

    // Should NOT throw
    await useAuthStore.getState().logout()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(localStorage.getItem('alphaforge-token')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// initAuth()
// ---------------------------------------------------------------------------
describe('initAuth()', () => {
  it('sets isLoading to false when no stored token exists', async () => {
    // localStorage is clean (afterEach clears it)
    useAuthStore.setState({ isLoading: true })

    await useAuthStore.getState().initAuth()

    const state = useAuthStore.getState()
    expect(state.isLoading).toBe(false)
    expect(state.user).toBeNull()
  })

  it('fetches user profile with valid stored token', async () => {
    localStorage.setItem('alphaforge-token', 'valid-token')

    server.use(
      http.get('/api/v1/admin/auth/me', () => {
        return HttpResponse.json(mockUser)
      })
    )

    await useAuthStore.getState().initAuth()

    const state = useAuthStore.getState()
    expect(state.user).toEqual(mockUser)
    expect(state.token).toBe('valid-token')
    expect(state.isLoading).toBe(false)
  })

  it('clears tokens and nulls state when getMe fails (expired token)', async () => {
    localStorage.setItem('alphaforge-token', 'expired-token')
    localStorage.setItem('alphaforge-refresh', 'some-refresh')

    server.use(
      http.get('/api/v1/admin/auth/me', () => {
        return new HttpResponse(null, { status: 401 })
      }),
      // The 401 interceptor in client.ts will try to refresh.
      // We make that fail too so initAuth's catch branch executes.
      http.post('/api/v1/admin/auth/refresh', () => {
        return new HttpResponse(null, { status: 401 })
      })
    )

    await useAuthStore.getState().initAuth()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(state.isLoading).toBe(false)
    expect(localStorage.getItem('alphaforge-token')).toBeNull()
    expect(localStorage.getItem('alphaforge-refresh')).toBeNull()
  })

  it('sets isLoading false even on network error', async () => {
    localStorage.setItem('alphaforge-token', 'some-token')

    server.use(
      http.get('/api/v1/admin/auth/me', () => {
        return HttpResponse.error()
      })
    )

    await useAuthStore.getState().initAuth()

    const state = useAuthStore.getState()
    expect(state.isLoading).toBe(false)
    expect(state.user).toBeNull()
  })
})
