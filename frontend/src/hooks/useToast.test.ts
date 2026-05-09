import { renderHook, act } from '@/test/test-utils'
import { reducer } from './useToast'
import type { ReactNode } from 'react'

// ---------------------------------------------------------------------------
// We need to isolate module-level mutable state between tests.
// Re-import the module fresh for each test via dynamic import after resetModules.
// ---------------------------------------------------------------------------

// Type shortcuts extracted from source
interface ToasterToast {
  id: string
  title?: React.ReactNode
  description?: React.ReactNode
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

interface State {
  toasts: ToasterToast[]
}

// ---------------------------------------------------------------------------
// Reducer tests (pure function — no module state issues)
// ---------------------------------------------------------------------------
describe('reducer', () => {
  it('ADD_TOAST adds a toast to an empty list', () => {
    const state: State = { toasts: [] }
    const toast: ToasterToast = { id: '1', title: 'Hello' }

    const next = reducer(state, { type: 'ADD_TOAST', toast: toast as never })
    expect(next.toasts).toHaveLength(1)
    expect(next.toasts[0].id).toBe('1')
    expect(next.toasts[0].title).toBe('Hello')
  })

  it('ADD_TOAST respects TOAST_LIMIT of 1 — only keeps the newest', () => {
    const state: State = {
      toasts: [{ id: '1', title: 'First' } as never],
    }
    const toast: ToasterToast = { id: '2', title: 'Second' }

    const next = reducer(state, { type: 'ADD_TOAST', toast: toast as never })
    // TOAST_LIMIT is 1, so only the newest toast (prepended) survives
    expect(next.toasts).toHaveLength(1)
    expect(next.toasts[0].id).toBe('2')
  })

  it('UPDATE_TOAST updates the matching toast by id', () => {
    const state: State = {
      toasts: [{ id: '1', title: 'Old', description: 'Desc' } as never],
    }

    const next = reducer(state, {
      type: 'UPDATE_TOAST',
      toast: { id: '1', title: 'New' },
    })
    expect(next.toasts[0].title).toBe('New')
    expect(next.toasts[0].description).toBe('Desc')
  })

  it('UPDATE_TOAST does not modify other toasts', () => {
    // Even though TOAST_LIMIT is 1, we test the reducer in isolation
    const state: State = {
      toasts: [
        { id: '1', title: 'A' } as never,
      ],
    }

    const next = reducer(state, {
      type: 'UPDATE_TOAST',
      toast: { id: '99', title: 'Nope' },
    })
    expect(next.toasts[0].title).toBe('A')
  })

  it('DISMISS_TOAST sets open:false on the target toast', () => {
    const state: State = {
      toasts: [{ id: '1', title: 'A', open: true } as never],
    }

    const next = reducer(state, { type: 'DISMISS_TOAST', toastId: '1' })
    expect(next.toasts[0].open).toBe(false)
  })

  it('DISMISS_TOAST with no toastId sets open:false on all toasts', () => {
    const state: State = {
      toasts: [{ id: '1', open: true } as never],
    }

    const next = reducer(state, { type: 'DISMISS_TOAST' })
    expect(next.toasts.every((t) => t.open === false)).toBe(true)
  })

  it('REMOVE_TOAST removes the toast with matching id', () => {
    const state: State = {
      toasts: [{ id: '1', title: 'A' } as never],
    }

    const next = reducer(state, { type: 'REMOVE_TOAST', toastId: '1' })
    expect(next.toasts).toHaveLength(0)
  })

  it('REMOVE_TOAST with no toastId clears all toasts', () => {
    const state: State = {
      toasts: [{ id: '1' } as never],
    }

    const next = reducer(state, { type: 'REMOVE_TOAST' })
    expect(next.toasts).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Hook integration tests — use dynamic imports to reset module-level state
// ---------------------------------------------------------------------------
describe('useToast hook', () => {
  // We need to reset module state before each test
  let useToastFn: () => {
    toasts: ToasterToast[]
    toast: (props: { title?: string; description?: string }) => {
      id: string
      dismiss: () => void
      update: (props: ToasterToast) => void
    }
    dismiss: (toastId?: string) => void
  }
  let toastFn: (props: { title?: string; description?: string }) => {
    id: string
    dismiss: () => void
    update: (props: ToasterToast) => void
  }

  beforeEach(async () => {
    vi.resetModules()
    const mod = await import('./useToast')
    useToastFn = mod.useToast as typeof useToastFn
    toastFn = mod.toast as typeof toastFn
  })

  it('returns empty toasts array initially', () => {
    const { result } = renderHook(() => useToastFn())
    expect(result.current.toasts).toEqual([])
  })

  it('toast() adds a toast and the hook reflects it', () => {
    const { result } = renderHook(() => useToastFn())

    act(() => {
      toastFn({ title: 'Hello' })
    })

    expect(result.current.toasts).toHaveLength(1)
    expect(result.current.toasts[0].title).toBe('Hello')
    expect(result.current.toasts[0].open).toBe(true)
  })

  it('toast() returns an object with id, dismiss, and update', () => {
    const { result } = renderHook(() => useToastFn())

    let returned: { id: string; dismiss: () => void; update: (props: ToasterToast) => void }
    act(() => {
      returned = toastFn({ title: 'Test' })
    })

    expect(returned!.id).toBeDefined()
    expect(typeof returned!.dismiss).toBe('function')
    expect(typeof returned!.update).toBe('function')
  })

  it('dismiss(toastId) sets open:false on a specific toast', () => {
    const { result } = renderHook(() => useToastFn())

    let toastResult: { id: string; dismiss: () => void }
    act(() => {
      toastResult = toastFn({ title: 'Dismissable' })
    })

    expect(result.current.toasts[0].open).toBe(true)

    act(() => {
      result.current.dismiss(toastResult!.id)
    })

    expect(result.current.toasts[0].open).toBe(false)
  })

  it('dismiss() with no id dismisses all toasts', () => {
    const { result } = renderHook(() => useToastFn())

    act(() => {
      toastFn({ title: 'One' })
    })

    act(() => {
      result.current.dismiss()
    })

    expect(result.current.toasts.every((t) => t.open === false)).toBe(true)
  })

  it('TOAST_LIMIT of 1 means only the latest toast survives', () => {
    const { result } = renderHook(() => useToastFn())

    act(() => {
      toastFn({ title: 'First' })
    })
    act(() => {
      toastFn({ title: 'Second' })
    })

    expect(result.current.toasts).toHaveLength(1)
    expect(result.current.toasts[0].title).toBe('Second')
  })
})
