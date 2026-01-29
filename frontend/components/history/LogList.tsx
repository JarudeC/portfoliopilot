'use client'

import { useState } from 'react'
import { LazyTrainingLog } from '@/lib/types/training'
import LogItem from './LogItem'
import DeleteConfirmModal from '@/components/ui/DeleteConfirmModal'

interface LogListProps {
  logs: LazyTrainingLog[]
  loading: boolean
  hasMore: boolean
  onLoadMore: () => void
  onDeleteLog: (logId: string) => void
  onBulkDelete: (logIds: string[]) => void
  onDeleteAll: () => void
}

export default function LogList({ logs, loading, hasMore, onLoadMore, onDeleteLog, onBulkDelete, onDeleteAll }: LogListProps) {
  const [expandedLog, setExpandedLog] = useState<string | null>(null)
  const [selectedLogs, setSelectedLogs] = useState<Set<string>>(new Set())
  const [isSelectMode, setIsSelectMode] = useState(false)
  const [showBulkDeleteModal, setShowBulkDeleteModal] = useState(false)
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const toggleExpanded = (logId: string) => {
    setExpandedLog(expandedLog === logId ? null : logId)
  }

  const toggleSelectMode = () => {
    setIsSelectMode(!isSelectMode)
    setSelectedLogs(new Set())
  }

  const toggleLogSelection = (logId: string) => {
    setSelectedLogs(prev => {
      const newSet = new Set(prev)
      if (newSet.has(logId)) {
        newSet.delete(logId)
      } else {
        newSet.add(logId)
      }
      return newSet
    })
  }

  const selectAll = () => {
    setSelectedLogs(new Set(logs.map(log => log.id)))
  }

  const clearSelection = () => {
    setSelectedLogs(new Set())
  }

  const handleBulkDeleteClick = () => {
    if (selectedLogs.size > 0) {
      setShowBulkDeleteModal(true)
    }
  }

  const handleDeleteAllClick = () => {
    setShowDeleteAllModal(true)
  }

  const handleBulkDeleteConfirm = async () => {
    setIsDeleting(true)
    try {
      await onBulkDelete(Array.from(selectedLogs))
      setSelectedLogs(new Set())
      setIsSelectMode(false)
      setShowBulkDeleteModal(false)
    } catch (error) {
      console.error('Bulk delete failed:', error)
    } finally {
      setIsDeleting(false)
    }
  }

  const handleDeleteAllConfirm = async () => {
    setIsDeleting(true)
    try {
      await onDeleteAll()
      setSelectedLogs(new Set())
      setIsSelectMode(false)
      setShowDeleteAllModal(false)
    } catch (error) {
      console.error('Delete all failed:', error)
    } finally {
      setIsDeleting(false)
    }
  }

  const handleModalCancel = () => {
    if (!isDeleting) {
      setShowBulkDeleteModal(false)
      setShowDeleteAllModal(false)
    }
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
          {isSelectMode && selectedLogs.size > 0 && (
            <span className="ml-2 text-sm text-[#4CC9F0]">
              ({selectedLogs.size} selected)
            </span>
          )}
        </h2>
        
        <div className="flex items-center gap-2">
          {!isSelectMode ? (
            <button
              onClick={toggleSelectMode}
              className="bg-[#1B263B] hover:bg-[#14273F] text-white px-4 py-2 rounded-lg transition text-sm"
            >
              Select Items
            </button>
          ) : (
            <>
              <button
                onClick={selectAll}
                className="bg-[#1B263B] hover:bg-[#14273F] text-white px-3 py-2 rounded-lg transition text-sm"
              >
                Select All
              </button>
              <button
                onClick={clearSelection}
                className="bg-[#1B263B] hover:bg-[#14273F] text-white px-3 py-2 rounded-lg transition text-sm"
              >
                Clear
              </button>
              <button
                onClick={handleBulkDeleteClick}
                disabled={selectedLogs.size === 0}
                className="bg-red-600 hover:bg-red-700 disabled:bg-red-600/50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition text-sm"
              >
                Delete Selected ({selectedLogs.size})
              </button>
              <button
                onClick={handleDeleteAllClick}
                className="bg-red-700 hover:bg-red-800 text-white px-4 py-2 rounded-lg transition text-sm"
              >
                Delete All
              </button>
              <button
                onClick={toggleSelectMode}
                className="bg-gray-600 hover:bg-gray-700 text-white px-3 py-2 rounded-lg transition text-sm"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {logs.map((log) => (
          <LogItem
            key={log.id}
            log={log}
            isExpanded={expandedLog === log.id}
            onToggleExpanded={() => toggleExpanded(log.id)}
            onDelete={() => onDeleteLog(log.id)}
            isSelectMode={isSelectMode}
            isSelected={selectedLogs.has(log.id)}
            onToggleSelect={() => toggleLogSelection(log.id)}
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

      {/* Bulk Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={showBulkDeleteModal}
        onClose={handleModalCancel}
        onConfirm={handleBulkDeleteConfirm}
        title="Delete Selected Training Logs"
        message={`Are you sure you want to delete ${selectedLogs.size} selected training log${selectedLogs.size > 1 ? 's' : ''}? This will permanently remove all associated data including results, charts, and parameters.`}
        itemName={`${selectedLogs.size} Training Log${selectedLogs.size > 1 ? 's' : ''}`}
        loading={isDeleting}
      />

      {/* Delete All Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={showDeleteAllModal}
        onClose={handleModalCancel}
        onConfirm={handleDeleteAllConfirm}
        title="Delete All Training Logs"
        message={`Are you sure you want to delete ALL ${logs.length} training logs? This will permanently remove all associated data including results, charts, and parameters. This action cannot be undone.`}
        itemName={`All ${logs.length} Training Logs`}
        loading={isDeleting}
      />
    </div>
  )
}