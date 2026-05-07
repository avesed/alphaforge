import { useQuery } from '@tanstack/react-query'
import type { AxiosResponse } from 'axios'
import type { TaskStatus } from '@/types'

interface UseTaskPollingOptions {
  interval?: number
  onComplete?: (data: TaskStatus) => void
  onError?: (data: TaskStatus) => void
}

export function useTaskPolling(
  taskId: string | null,
  fetcher: (id: string) => Promise<AxiosResponse<TaskStatus>>,
  options?: UseTaskPollingOptions
) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: async () => {
      const response = await fetcher(taskId!)
      const data = response.data

      if (data.status === 'completed') {
        options?.onComplete?.(data)
      } else if (data.status === 'failed') {
        options?.onError?.(data)
      }

      return data
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed') return false
      return options?.interval ?? 3000
    },
  })
}
