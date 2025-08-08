'use client'

import { useState } from 'react'
import { TrainingLog } from '@/lib/types/training'
import LogDetails from './LogDetails'
import DeleteConfirmModal from '../../../components/ui/DeleteConfirmModal'

interface LogItemProps {
  log: TrainingLog
  isExpanded: boolean
  onToggleExpanded: () => void
  onDelete: () => void
  isSelectMode?: boolean
  isSelected?: boolean
  onToggleSelect?: () => void
}

export default function LogItem({ 
  log, 
  isExpanded, 
  onToggleExpanded, 
  onDelete, 
  isSelectMode = false, 
  isSelected = false, 
  onToggleSelect 
}: LogItemProps) {
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getTypeColor = (type: string) => {
    return type === 'forecast' ? 'text-[#4CC9F0]' : 'text-[#3A86FF]'
  }

  const getTypeIcon = (type: string) => {
    return type === 'forecast' ? '📈' : '🔄'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400'
      case 'failed': return 'text-red-400' 
      case 'in_progress': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowDeleteModal(true)
  }

  const handleDeleteConfirm = async () => {
    setIsDeleting(true)
    try {
      await onDelete()
      setShowDeleteModal(false)
    } catch (error) {
      console.error('Delete failed:', error)
    } finally {
      setIsDeleting(false)
    }
  }

  const handleDeleteCancel = () => {
    if (!isDeleting) {
      setShowDeleteModal(false)
    }
  }

  const handleCheckboxClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (onToggleSelect) {
      onToggleSelect()
    }
  }

  const handleRowClick = () => {
    if (isSelectMode && onToggleSelect) {
      onToggleSelect()
    } else {
      onToggleExpanded()
    }
  }

  return (
    <div className={`bg-[#0d1b2a]/50 rounded-lg border overflow-hidden transition-colors ${
      isSelected ? 'border-[#4CC9F0] bg-[#4CC9F0]/10' : 'border-[#1B263B]'
    }`}>
      <div
        className="p-4 cursor-pointer hover:bg-[#14273F]/50 transition-colors"
        onClick={handleRowClick}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isSelectMode && (
              <div onClick={handleCheckboxClick}>
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => {}} // Controlled by parent
                  className="w-4 h-4 rounded border-gray-300 text-[#4CC9F0] focus:ring-[#4CC9F0] focus:ring-2 cursor-pointer"
                />
              </div>
            )}
            <span className="text-xl">{getTypeIcon(log.type)}</span>
            
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className={`font-semibold capitalize ${getTypeColor(log.type)}`}>
                  {log.type}
                </span>
                <span className="text-gray-400">•</span>
                <span className="text-white font-medium">{log.model}</span>
              </div>
              
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span>{log.stocks.length} stock{log.stocks.length !== 1 ? 's' : ''}</span>
                <span>•</span>
                <span>{formatDate(log.created_at)}</span>
                <span>•</span>
                <span className={getStatusColor(log.status)}>{log.status}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!isSelectMode && (
              <button
                onClick={handleDeleteClick}
                className="p-2 text-gray-400 hover:text-red-400 transition"
                title="Delete log"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
            
            {!isSelectMode && (
              <svg
                className={`h-4 w-4 shrink-0 transition-transform duration-200 ${
                  isExpanded ? 'rotate-180' : ''
                }`}
                viewBox="0 0 24 24"
                stroke="currentColor"
                fill="none"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            )}
          </div>
        </div>

        <div className="flex flex-wrap gap-1 mt-3">
          {log.stocks.slice(0, 6).map((stock) => (
            <span
              key={stock}
              className="text-xs bg-[#1B263B] text-cyan-300 px-2 py-1 rounded"
            >
              {stock}
            </span>
          ))}
          {log.stocks.length > 6 && (
            <span className="text-xs text-gray-400 px-2 py-1">
              +{log.stocks.length - 6} more
            </span>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-[#1B263B]">
          <LogDetails log={log} />
        </div>
      )}

      <DeleteConfirmModal
        isOpen={showDeleteModal}
        onClose={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
        title="Delete Training Log"
        message="Are you sure you want to delete this training log? This will permanently remove all associated data including results, charts, and parameters."
        itemName={`${log.type === 'forecast' ? 'Forecast' : 'Backtest'} • ${log.model} • ${log.stocks.join(', ')}`}
        loading={isDeleting}
      />
    </div>
  )
}