import { http, HttpResponse } from 'msw'
import { server } from '@/test/mocks/server'
import { render, screen, waitFor } from '@/test/test-utils'
import { useAuthStore } from '@/stores/authStore'
import DashboardPage from './DashboardPage'
import type { DashboardStats, PredictionModel } from '@/types'

// Initialize i18n so useTranslation works
import '@/i18n'

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
const mockStats: DashboardStats = {
  userCount: 5,
  consumerCount: 3,
  schedulerIsLeader: true,
  schedulerJobCount: 8,
  stockpulseConnected: true,
  redisConnected: true,
}

const mockModels: PredictionModel[] = [
  {
    id: 'model-us-1',
    market: 'us',
    modelDate: '2026-05-07',
    modelType: 'ranking',
    forwardDays: 5,
    ic: 0.035,
    icir: 0.28,
    ndcg: 0.65,
    qualityPassed: true,
    featureCount: 120,
    symbolCount: 500,
    createdAt: '2026-05-07T10:00:00Z',
  },
  {
    id: 'model-cn-1',
    market: 'cn',
    modelDate: '2026-05-07',
    modelType: 'ranking',
    forwardDays: 5,
    ic: 0.022,
    icir: 0.15,
    ndcg: 0.58,
    qualityPassed: true,
    featureCount: 80,
    symbolCount: 300,
    createdAt: '2026-05-07T09:00:00Z',
  },
  {
    id: 'model-hk-1',
    market: 'hk',
    modelDate: '2026-05-06',
    modelType: 'ranking',
    forwardDays: 5,
    ic: 0.008,
    icir: 0.05,
    ndcg: 0.52,
    qualityPassed: false,
    featureCount: 60,
    symbolCount: 200,
    createdAt: '2026-05-06T11:00:00Z',
  },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setupSuccessHandlers() {
  server.use(
    http.get('/api/v1/admin/stats', () => {
      return HttpResponse.json(mockStats)
    }),
    http.get('/api/v1/predictions/models', () => {
      return HttpResponse.json(mockModels)
    })
  )
}

function setupErrorHandlers() {
  server.use(
    http.get('/api/v1/admin/stats', () => {
      return HttpResponse.json({ detail: 'Internal error' }, { status: 500 })
    }),
    http.get('/api/v1/predictions/models', () => {
      return HttpResponse.json({ detail: 'Internal error' }, { status: 500 })
    })
  )
}

// ---------------------------------------------------------------------------
// Reset between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  useAuthStore.setState({
    user: {
      id: 1,
      email: 'admin@alphaforge.dev',
      displayName: 'Admin',
      role: 'admin',
      locale: 'en',
    },
    token: 'test-token',
    isLoading: false,
  })
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('DashboardPage', () => {
  it('renders the dashboard title', () => {
    setupSuccessHandlers()
    render(<DashboardPage />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
  })

  it('renders loading skeletons initially', () => {
    // Use a handler that delays so we catch the loading state
    server.use(
      http.get('/api/v1/admin/stats', async () => {
        await new Promise((r) => setTimeout(r, 5000))
        return HttpResponse.json(mockStats)
      }),
      http.get('/api/v1/predictions/models', async () => {
        await new Promise((r) => setTimeout(r, 5000))
        return HttpResponse.json(mockModels)
      })
    )

    const { container } = render(<DashboardPage />)

    // Skeleton elements should be present (animate-pulse class)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders market cards with data after loading', async () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    // Wait for the market labels to appear
    await waitFor(() => {
      expect(screen.getByText('US Market')).toBeInTheDocument()
    })
    expect(screen.getByText('China A-Share')).toBeInTheDocument()
    expect(screen.getByText('Hong Kong')).toBeInTheDocument()
  })

  it('displays IC values from the model data', async () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    await waitFor(() => {
      // US market IC = 0.035 -> formatted as "0.0350"
      expect(screen.getByText('0.0350')).toBeInTheDocument()
    })
  })

  it('shows system status with connection indicators', async () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    // Wait for the stats data to load and "Connected" to appear
    await waitFor(() => {
      const connectedElements = screen.getAllByText('Connected')
      expect(connectedElements.length).toBe(2) // StockPulse + Redis
    })
  })

  it('shows scheduler leader badge when schedulerIsLeader is true', async () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText('Leader')).toBeInTheDocument()
    })

    expect(screen.getByText('8 jobs')).toBeInTheDocument()
  })

  it('renders error message when models query fails', async () => {
    setupErrorHandlers()
    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText(/failed to load model data/i)).toBeInTheDocument()
    })
  })

  it('renders Quick Actions section with trigger buttons for each market', async () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    await waitFor(() => {
      expect(screen.getByText('Quick Actions')).toBeInTheDocument()
    })

    expect(screen.getByText('Run China A-Share')).toBeInTheDocument()
    expect(screen.getByText('Run US Market')).toBeInTheDocument()
    expect(screen.getByText('Run Hong Kong')).toBeInTheDocument()
  })

  it('renders a refresh button', () => {
    setupSuccessHandlers()
    render(<DashboardPage />)

    expect(screen.getByText('Refresh')).toBeInTheDocument()
  })
})
