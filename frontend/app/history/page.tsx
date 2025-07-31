'use client'

import { useEffect, useState } from 'react'
import { useAuth } from '@/contexts/AuthContexts'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import LogList from './components/LogList'
import LogFilters from './components/LogFilters'
import { TrainingLog } from '@/lib/types/training'

export default function HistoryPage() {
  const { user, loading: authLoading } = useAuth()
  const router = useRouter()
  const [logs, setLogs] = useState<TrainingLog[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    type: 'all' as 'all' | 'forecast' | 'backtest',
    search: '',
    model: '',
    dateFrom: '',
    dateTo: ''
  })
  const [page, setPage] = useState(0)
  const [hasMore, setHasMore] = useState(true)

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/')
      return
    }
  }, [user, authLoading, router])

  const fetchLogs = async (reset = false) => {
    if (!user) return

    try {
      setLoading(true)
      setError(null)

      const currentPage = reset ? 0 : page
      const params = new URLSearchParams({
        limit: '20',
        offset: (currentPage * 20).toString()
      })

      if (filters.type !== 'all') {
        params.set('type', filters.type)
      }

      const res = await fetch(`/api/train/history?${params}`)
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.error || 'Failed to fetch training history')
      }

      const data = await res.json()
      
      if (reset) {
        setLogs(data.logs)
        setPage(0)
      } else {
        setLogs(prev => [...prev, ...data.logs])
      }
      
      setHasMore(data.logs.length === 20)
      if (!reset) {
        setPage(prev => prev + 1)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (user) {
      fetchLogs(true)
    }
  }, [user, filters.type])

  const filteredLogs = logs.filter(log => {
    if (filters.search && !log.stocks.some(stock => 
      stock.toLowerCase().includes(filters.search.toLowerCase())
    ) && !log.model.toLowerCase().includes(filters.search.toLowerCase())) {
      return false
    }
    
    if (filters.model && log.model !== filters.model) {
      return false
    }
    
    if (filters.dateFrom && new Date(log.created_at) < new Date(filters.dateFrom)) {
      return false
    }
    
    if (filters.dateTo && new Date(log.created_at) > new Date(filters.dateTo)) {
      return false
    }
    
    return true
  })

  const handleDeleteLog = async (logId: string) => {
    try {
      const res = await fetch(`/api/train/${logId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Failed to delete log')
      
      setLogs(prev => prev.filter(log => log.id !== logId))
    } catch (err) {
      alert('Failed to delete training log')
    }
  }

  if (authLoading || !user) {
    return (
      <div className="min-h-screen bg-[#0D1B2A] flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0D1B2A] text-white flex flex-col">
      <Navbar />
      
      <main className="flex-1 pt-[72px] pb-20 px-4 lg:px-16">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Training History</h1>
            <p className="text-gray-400">
              View and manage your past forecasts and backtests
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
            <aside className="bg-[#14273F] rounded-xl p-6">
              <LogFilters 
                filters={filters}
                onFiltersChange={setFilters}
                availableModels={Array.from(new Set(logs.map(log => log.model)))}
              />
            </aside>

            <section className="bg-[#14273F] rounded-xl p-6">
              {error ? (
                <div className="text-center py-8">
                  <p className="text-red-400 mb-4">{error}</p>
                  <button 
                    onClick={() => fetchLogs(true)}
                    className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-4 py-2 rounded-full transition"
                  >
                    Retry
                  </button>
                </div>
              ) : (
                <LogList
                  logs={filteredLogs}
                  loading={loading}
                  hasMore={hasMore}
                  onLoadMore={() => fetchLogs(false)}
                  onDeleteLog={handleDeleteLog}
                />
              )}
            </section>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}