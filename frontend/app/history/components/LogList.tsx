'use client'

import { useState } from 'react'
import { TrainingLog } from '@/lib/types/training'
import LogItem from './LogItem'

interface LogListProps {
  logs: TrainingLog[]
  loading: boolean
  hasMore: boolean
  onLoadMore: () => void
  onDeleteLog: (logId: string) => void
}

export default function LogList({ logs, loading, hasMore, onLoadMore, onDeleteLog }: LogListProps) {
  const [expandedLog, setExpandedLog] = useState<string | null>(null)

  const toggleExpanded = (logId: string) => {
    setExpandedLog(expandedLog === logId ? null : logId)
  }

  if (loading && logs.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-400">Loading training history...</div>
      </div>
    )
  }

  if (logs.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">📊</div>
        <h3 className="text-lg font-semibold mb-2">No training history</h3>
        <p className="text-gray-400 mb-6">
          Run some forecasts or backtests to see your results here
        </p>
        <a
          href="/dashboard"
          className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-6 py-2 rounded-full transition inline-block"
        >
          Start Training
        </a>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold">
          {logs.length} Training{logs.length !== 1 ? 's' : ''} Found
        </h2>
      </div>

      <div className="space-y-3">
        {logs.map((log) => (
          <LogItem
            key={log.id}
            log={log}
            isExpanded={expandedLog === log.id}
            onToggleExpanded={() => toggleExpanded(log.id)}
            onDelete={() => onDeleteLog(log.id)}
          />
        ))}
      </div>

      {hasMore && (
        <div className="flex justify-center pt-6">
          <button
            onClick={onLoadMore}
            disabled={loading}
            className="bg-[#1B263B] hover:bg-[#14273F] text-white px-6 py-2 rounded-full transition disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}
    </div>
  )
}