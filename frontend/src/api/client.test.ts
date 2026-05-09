import { http, HttpResponse } from 'msw'
import { server } from '@/test/mocks/server'
import apiClient, {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
  getErrorMessage,
} from './client'
import axios from 'axios'

// ---------------------------------------------------------------------------
// Helper: spy on window.location.href assignments
// ---------------------------------------------------------------------------
let locationAssignments: string[] = []
const originalLocation = window.location

beforeEach(() => {
  locationAssignments = []
  // Preserve jsdom's real location (with origin etc.) but intercept href set
  const locationProxy = new Proxy(originalLocation, {
    set(_target, prop, value) {
      if (prop === 'href') {
        locationAssignments.push(value as string)
        return true
      }
      return Reflect.set(_target, prop, value)
    },
    get(target, prop) {
      const val = Reflect.get(target, prop)
      if (typeof val === 'function') {
        return val.bind(target)
      }
      return val
    },
  })
  Object.defineProperty(window, 'location', {
    writable: true,
    value: locationProxy,
  })
})

afterEach(() => {
  // Restore original location so other test files are unaffected
  Object.defineProperty(window, 'location', {
    writable: true,
    value: originalLocation,
  })
})

// ---------------------------------------------------------------------------
// getAccessToken()
// ---------------------------------------------------------------------------
describe('getAccessToken()', () => {
  it('returns null when no token is stored', () => {
    expect(getAccessToken()).toBeNull()
  })

  it('returns the stored access token', () => {
    localStorage.setItem('alphaforge-token', 'abc123')
    expect(getAccessToken()).toBe('abc123')
  })
})

// ---------------------------------------------------------------------------
// getRefreshToken()
// ---------------------------------------------------------------------------
describe('getRefreshToken()', () => {
  it('returns null when no refresh token is stored', () => {
    expect(getRefreshToken()).toBeNull()
  })

  it('returns the stored refresh token', () => {
    localStorage.setItem('alphaforge-refresh', 'refresh-xyz')
    expect(getRefreshToken()).toBe('refresh-xyz')
  })
})

// ---------------------------------------------------------------------------
// setTokens()
// ---------------------------------------------------------------------------
describe('setTokens()', () => {
  it('writes both tokens to localStorage', () => {
    setTokens('access-1', 'refresh-1')
    expect(localStorage.getItem('alphaforge-token')).toBe('access-1')
    expect(localStorage.getItem('alphaforge-refresh')).toBe('refresh-1')
  })

  it('overwrites previously stored tokens', () => {
    setTokens('old-access', 'old-refresh')
    setTokens('new-access', 'new-refresh')
    expect(localStorage.getItem('alphaforge-token')).toBe('new-access')
    expect(localStorage.getItem('alphaforge-refresh')).toBe('new-refresh')
  })
})

// ---------------------------------------------------------------------------
// clearTokens()
// ---------------------------------------------------------------------------
describe('clearTokens()', () => {
  it('removes both tokens from localStorage', () => {
    setTokens('a', 'r')
    clearTokens()
    expect(localStorage.getItem('alphaforge-token')).toBeNull()
    expect(localStorage.getItem('alphaforge-refresh')).toBeNull()
  })

  it('is safe to call when no tokens exist', () => {
    expect(() => clearTokens()).not.toThrow()
  })
})

// ---------------------------------------------------------------------------
// getErrorMessage()
// ---------------------------------------------------------------------------
describe('getErrorMessage()', () => {
  it('extracts detail from an AxiosError response', () => {
    const error = new axios.AxiosError(
      'Request failed',
      '400',
      undefined,
      undefined,
      {
        data: { detail: 'Invalid credentials' },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: {} as ReturnType<typeof axios.create>['defaults'],
      } as never
    )
    expect(getErrorMessage(error)).toBe('Invalid credentials')
  })

  it('falls back to error.message when no detail in response', () => {
    const error = new axios.AxiosError('Network Error')
    expect(getErrorMessage(error)).toBe('Network Error')
  })

  it('handles a plain Error', () => {
    expect(getErrorMessage(new Error('Something broke'))).toBe('Something broke')
  })

  it('returns generic string for non-Error values', () => {
    expect(getErrorMessage('oops')).toBe('An unexpected error occurred')
  })

  it('returns generic string for null', () => {
    expect(getErrorMessage(null)).toBe('An unexpected error occurred')
  })
})

// ---------------------------------------------------------------------------
// Request interceptor
// ---------------------------------------------------------------------------
describe('Request interceptor', () => {
  it('attaches Authorization header when token exists', async () => {
    let capturedAuth: string | undefined
    server.use(
      http.get('http://localhost:3000/api/v1/test-auth', ({ request }) => {
        capturedAuth = request.headers.get('authorization') ?? undefined
        return HttpResponse.json({ ok: true })
      })
    )

    setTokens('my-token', 'my-refresh')
    await apiClient.get('/test-auth')

    expect(capturedAuth).toBe('Bearer my-token')
  })

  it('does NOT attach Authorization header when no token exists', async () => {
    let capturedAuth: string | undefined | null
    server.use(
      http.get('http://localhost:3000/api/v1/test-no-auth', ({ request }) => {
        capturedAuth = request.headers.get('authorization')
        return HttpResponse.json({ ok: true })
      })
    )

    // localStorage is clean (no token)
    await apiClient.get('/test-no-auth')

    expect(capturedAuth).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Response interceptor — 401 handling
// ---------------------------------------------------------------------------
describe('Response interceptor — 401 refresh flow', () => {
  it('refreshes token and retries original request on 401', async () => {
    let callCount = 0

    server.use(
      http.get('http://localhost:3000/api/v1/protected-data', ({ request }) => {
        callCount++
        const auth = request.headers.get('authorization')
        if (auth === 'Bearer refreshed-token') {
          return HttpResponse.json({ data: 'secret' })
        }
        return new HttpResponse(null, { status: 401 })
      }),
      http.post('http://localhost:3000/api/v1/admin/auth/refresh', () => {
        return HttpResponse.json({
          accessToken: 'refreshed-token',
          refreshToken: 'new-refresh',
        })
      })
    )

    setTokens('expired-token', 'valid-refresh')
    const response = await apiClient.get('/protected-data')

    expect(response.data).toEqual({ data: 'secret' })
    // First call fails with 401, then refresh, then retry
    expect(callCount).toBe(2)
    // Tokens updated in localStorage
    expect(localStorage.getItem('alphaforge-token')).toBe('refreshed-token')
    expect(localStorage.getItem('alphaforge-refresh')).toBe('new-refresh')
  })

  it('clears tokens and redirects to /login when refresh endpoint itself returns 401', async () => {
    server.use(
      http.get('http://localhost:3000/api/v1/some-endpoint', () => {
        return new HttpResponse(null, { status: 401 })
      }),
      http.post('http://localhost:3000/api/v1/admin/auth/refresh', () => {
        return new HttpResponse(null, { status: 401 })
      })
    )

    setTokens('bad-token', 'bad-refresh')

    await expect(apiClient.get('/some-endpoint')).rejects.toThrow()

    expect(localStorage.getItem('alphaforge-token')).toBeNull()
    expect(localStorage.getItem('alphaforge-refresh')).toBeNull()
    expect(locationAssignments).toContain('/login')
  })

  it('clears tokens and redirects when refresh call fails', async () => {
    server.use(
      http.get('http://localhost:3000/api/v1/guarded', () => {
        return new HttpResponse(null, { status: 401 })
      }),
      http.post('http://localhost:3000/api/v1/admin/auth/refresh', () => {
        return new HttpResponse(null, { status: 500 })
      })
    )

    setTokens('expired', 'some-refresh')

    await expect(apiClient.get('/guarded')).rejects.toThrow()

    expect(localStorage.getItem('alphaforge-token')).toBeNull()
    expect(localStorage.getItem('alphaforge-refresh')).toBeNull()
    expect(locationAssignments).toContain('/login')
  })

  it('does not retry on non-401 errors', async () => {
    let callCount = 0
    server.use(
      http.get('http://localhost:3000/api/v1/server-error', () => {
        callCount++
        return new HttpResponse(null, { status: 500 })
      })
    )

    await expect(apiClient.get('/server-error')).rejects.toThrow()
    expect(callCount).toBe(1)
  })

  it('deduplicates concurrent refresh calls (refreshPromise singleton)', async () => {
    let refreshCallCount = 0

    server.use(
      http.get('http://localhost:3000/api/v1/endpoint-a', ({ request }) => {
        const auth = request.headers.get('authorization')
        if (auth === 'Bearer concurrent-refreshed') {
          return HttpResponse.json({ a: true })
        }
        return new HttpResponse(null, { status: 401 })
      }),
      http.get('http://localhost:3000/api/v1/endpoint-b', ({ request }) => {
        const auth = request.headers.get('authorization')
        if (auth === 'Bearer concurrent-refreshed') {
          return HttpResponse.json({ b: true })
        }
        return new HttpResponse(null, { status: 401 })
      }),
      http.post('http://localhost:3000/api/v1/admin/auth/refresh', async () => {
        refreshCallCount++
        // Slight delay to allow both requests to hit refresh
        await new Promise((r) => setTimeout(r, 50))
        return HttpResponse.json({
          accessToken: 'concurrent-refreshed',
          refreshToken: 'concurrent-refresh-new',
        })
      })
    )

    setTokens('stale', 'valid-refresh')

    const [resA, resB] = await Promise.all([
      apiClient.get('/endpoint-a'),
      apiClient.get('/endpoint-b'),
    ])

    expect(resA.data).toEqual({ a: true })
    expect(resB.data).toEqual({ b: true })
    // Only one refresh call should have been made
    expect(refreshCallCount).toBe(1)
  })
})
