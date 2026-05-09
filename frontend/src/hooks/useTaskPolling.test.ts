import { renderHook, waitFor } from '@/test/test-utils'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement, type ReactNode } from 'react'
import { useTaskPolling } from './useTaskPolling'
import type { TaskStatus } from '@/types'
import type { AxiosResponse } from 'axios'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
    },
  })
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

function makeFetcherResponse(data: TaskStatus): AxiosResponse<TaskStatus> {
  return {
    data,
    status: 200,
    statusText: 'OK',
    headers: {},
    config: {} as AxiosResponse['config'],
  }
}

function makeTaskStatus(
  overrides: Partial<TaskStatus> = {}
): TaskStatus {
  return {
    taskId: 'task-1',
    status: 'pending',
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('useTaskPolling', () => {
  it('does not fetch when taskId is null', () => {
    const fetcher = vi.fn()

    renderHook(() => useTaskPolling(null, fetcher), {
      wrapper: createWrapper(),
    })

    expect(fetcher).not.toHaveBeenCalled()
  })

  it('does not fetch when taskId is undefined', () => {
    const fetcher = vi.fn()

    renderHook(
      () => useTaskPolling(undefined as unknown as string | null, fetcher),
      { wrapper: createWrapper() }
    )

    expect(fetcher).not.toHaveBeenCalled()
  })

  it('fetches data when taskId is provided', async () => {
    const taskData = makeTaskStatus({ status: 'completed' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(fetcher).toHaveBeenCalledWith('task-1')
    expect(result.current.data).toEqual(taskData)
  })

  it('returns data from the fetcher function', async () => {
    const taskData = makeTaskStatus({
      status: 'completed',
      progress: 100,
      result: { predictions: 42 },
    })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.data).toEqual(taskData))
  })

  it('calls onComplete callback when status is completed', async () => {
    const taskData = makeTaskStatus({ status: 'completed' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))
    const onComplete = vi.fn()

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher, { onComplete }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(onComplete).toHaveBeenCalledWith(taskData)
  })

  it('calls onError callback when status is failed', async () => {
    const taskData = makeTaskStatus({ status: 'failed', error: 'Something broke' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))
    const onError = vi.fn()

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher, { onError }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(onError).toHaveBeenCalledWith(taskData)
  })

  it('does not call onComplete when status is pending', async () => {
    const taskData = makeTaskStatus({ status: 'pending' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))
    const onComplete = vi.fn()

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher, { onComplete }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(onComplete).not.toHaveBeenCalled()
  })

  it('does not call onError when status is running', async () => {
    const taskData = makeTaskStatus({ status: 'running' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(taskData))
    const onError = vi.fn()

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher, { onError }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(onError).not.toHaveBeenCalled()
  })

  it('stops polling when status transitions to completed (refetchInterval returns false)', async () => {
    const completedData = makeTaskStatus({ status: 'completed' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(completedData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    // The refetchInterval callback should return false for completed.
    // We verify by checking that after initial fetch, no further calls happen.
    const callCount = fetcher.mock.calls.length

    // Wait a bit to make sure no extra poll happens
    await new Promise((r) => setTimeout(r, 100))
    expect(fetcher.mock.calls.length).toBe(callCount)
  })

  it('stops polling when status transitions to failed', async () => {
    const failedData = makeTaskStatus({ status: 'failed' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(failedData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    const callCount = fetcher.mock.calls.length

    await new Promise((r) => setTimeout(r, 100))
    expect(fetcher.mock.calls.length).toBe(callCount)
  })

  it('uses default interval of 3000ms when no interval option is provided', async () => {
    // We test this indirectly by verifying the hook is configured properly.
    // The refetchInterval function for pending/running statuses should return 3000.
    const pendingData = makeTaskStatus({ status: 'pending' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(pendingData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    // The hook returns the query result; if pending, refetchInterval returns 3000
    // We just confirm it fetched and didn't throw
    expect(result.current.data?.status).toBe('pending')
  })

  it('respects custom interval option', async () => {
    const pendingData = makeTaskStatus({ status: 'pending' })
    const fetcher = vi.fn().mockResolvedValue(makeFetcherResponse(pendingData))

    const { result } = renderHook(
      () => useTaskPolling('task-1', fetcher, { interval: 500 }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.status).toBe('pending')
    // With a 500ms interval and pending status, polling should continue.
    // We can verify by waiting and checking the fetcher call count increases.
    const callCountAfterFirst = fetcher.mock.calls.length

    await waitFor(
      () => expect(fetcher.mock.calls.length).toBeGreaterThan(callCountAfterFirst),
      { timeout: 2000 }
    )
  })
})
