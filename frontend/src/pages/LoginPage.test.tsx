import { http, HttpResponse } from 'msw'
import { server } from '@/test/mocks/server'
import { render, screen, userEvent, waitFor } from '@/test/test-utils'
import { useAuthStore } from '@/stores/authStore'
import { useThemeStore } from '@/stores/themeStore'
import LoginPage from './LoginPage'
import type { TokenResponse, User } from '@/types'

// Initialize i18n so useTranslation works
import '@/i18n'

// ---------------------------------------------------------------------------
// Mock react-router-dom's useNavigate
// ---------------------------------------------------------------------------
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
const mockUser: User = {
  id: 1,
  email: 'admin@alphaforge.dev',
  displayName: 'Admin',
  role: 'admin',
  locale: 'en',
}

const mockTokenResponse: TokenResponse = {
  accessToken: 'test-access-token',
  refreshToken: 'test-refresh-token',
  tokenType: 'bearer',
}

// ---------------------------------------------------------------------------
// Reset state between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  mockNavigate.mockClear()
  useAuthStore.setState({ user: null, token: null, isLoading: false })
  useThemeStore.setState({ theme: 'light', resolvedTheme: 'light' })
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('LoginPage', () => {
  it('renders email input', () => {
    render(<LoginPage />)
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
  })

  it('renders password input', () => {
    render(<LoginPage />)
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
  })

  it('renders the submit button', () => {
    render(<LoginPage />)
    const button = screen.getByRole('button', { name: /sign in/i })
    expect(button).toBeInTheDocument()
    expect(button).toBeEnabled()
  })

  it('renders the login title', () => {
    render(<LoginPage />)
    // The title is the app name "AlphaForge" per en.json login.title
    expect(screen.getByText('AlphaForge')).toBeInTheDocument()
  })

  it('renders the subtitle', () => {
    render(<LoginPage />)
    expect(screen.getByText('ML Stock Prediction Platform')).toBeInTheDocument()
  })

  it('allows typing in email and password fields', async () => {
    const user = userEvent.setup()
    render(<LoginPage />)

    const emailInput = screen.getByLabelText(/email/i)
    const passwordInput = screen.getByLabelText(/password/i)

    await user.type(emailInput, 'test@example.com')
    await user.type(passwordInput, 'password123')

    expect(emailInput).toHaveValue('test@example.com')
    expect(passwordInput).toHaveValue('password123')
  })

  it('navigates to "/" on successful login', async () => {
    server.use(
      http.post('/api/v1/admin/auth/login', () => {
        return HttpResponse.json(mockTokenResponse)
      }),
      http.get('/api/v1/admin/auth/me', () => {
        return HttpResponse.json(mockUser)
      })
    )

    const user = userEvent.setup()
    render(<LoginPage />)

    await user.type(screen.getByLabelText(/email/i), 'admin@alphaforge.dev')
    await user.type(screen.getByLabelText(/password/i), 'Admin123')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/')
    })
  })

  it('displays error message on failed login', async () => {
    server.use(
      http.post('/api/v1/admin/auth/login', () => {
        return HttpResponse.json(
          { detail: 'Invalid credentials' },
          { status: 401 }
        )
      }),
      // The 401 interceptor will try to refresh, make that fail too
      http.post('/api/v1/admin/auth/refresh', () => {
        return HttpResponse.json(
          { detail: 'No refresh' },
          { status: 401 }
        )
      })
    )

    const user = userEvent.setup()
    render(<LoginPage />)

    await user.type(screen.getByLabelText(/email/i), 'bad@email.com')
    await user.type(screen.getByLabelText(/password/i), 'wrong')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    await waitFor(() => {
      // The error message should appear on screen
      expect(screen.getByText(/invalid credentials|an unexpected error|request failed/i)).toBeInTheDocument()
    })

    // Should NOT navigate
    expect(mockNavigate).not.toHaveBeenCalledWith('/')
  })

  it('disables the submit button while submitting', async () => {
    // Use a delayed response to observe the loading state
    server.use(
      http.post('/api/v1/admin/auth/login', async () => {
        await new Promise((r) => setTimeout(r, 200))
        return HttpResponse.json(mockTokenResponse)
      }),
      http.get('/api/v1/admin/auth/me', () => {
        return HttpResponse.json(mockUser)
      })
    )

    const user = userEvent.setup()
    render(<LoginPage />)

    await user.type(screen.getByLabelText(/email/i), 'admin@alphaforge.dev')
    await user.type(screen.getByLabelText(/password/i), 'Admin123')

    const button = screen.getByRole('button', { name: /sign in/i })
    await user.click(button)

    // Button text should change to loading text while submitting
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /loading/i })).toBeDisabled()
    })

    // Eventually returns to normal after login completes
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/')
    })
  })

  it('has a theme toggle button', () => {
    render(<LoginPage />)
    // There should be a ghost button for theme toggle (it has an SVG icon)
    const buttons = screen.getAllByRole('button')
    // One submit button + one theme toggle button
    expect(buttons.length).toBeGreaterThanOrEqual(2)
  })
})
