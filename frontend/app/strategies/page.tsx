'use client'

import { useEffect, useState } from 'react'
import { useAuth } from '@/contexts/AuthContexts'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import DeleteConfirmModal from '@/components/ui/DeleteConfirmModal'
import EditStrategyModal from '@/components/ui/EditStrategyModal'
import { Trash2, Code, Calendar, Loader2, CheckSquare, Square, X, Pencil } from 'lucide-react'
import type { HydratedStrategy } from '@/lib/types/strategy'

export default function StrategiesPage() {
  const { user, loading: authLoading } = useAuth()
  const router = useRouter()
  const [strategies, setStrategies] = useState<HydratedStrategy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'backtest' | 'forecast'>('all')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<HydratedStrategy | null>(null)

  // Bulk selection state
  const [selectedStrategies, setSelectedStrategies] = useState<Set<string>>(new Set())
  const [isSelectMode, setIsSelectMode] = useState(false)
  const [showBulkDeleteModal, setShowBulkDeleteModal] = useState(false)
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false)
  const [bulkDeleting, setBulkDeleting] = useState(false)

  // Edit state
  const [editTarget, setEditTarget] = useState<HydratedStrategy | null>(null)

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/')
      return
    }
  }, [user, authLoading, router])

  const fetchStrategies = async () => {
    if (!user) return

    try {
      setLoading(true)
      setError(null)

      const params = filter !== 'all' ? `?mode=${filter}` : ''
      const res = await fetch(`/api/strategies${params}`)

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to fetch strategies')
      }

      const data = await res.json()
      setStrategies(data.strategies || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (user) {
      fetchStrategies()
    }
  }, [user, filter])

  const handleDeleteClick = (strategy: HydratedStrategy) => {
    setDeleteTarget(strategy)
  }

  const handleEditClick = (strategy: HydratedStrategy) => {
    setEditTarget(strategy)
  }

  const handleEditSave = (updated: HydratedStrategy) => {
    setStrategies(prev => prev.map(s => s.id === updated.id ? updated : s))
  }

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return

    try {
      setDeleting(deleteTarget.id)
      const res = await fetch(`/api/strategies/${deleteTarget.id}`, { method: 'DELETE' })

      if (!res.ok) {
        throw new Error('Failed to delete strategy')
      }

      setStrategies(prev => prev.filter(s => s.id !== deleteTarget.id))
      setSelectedStrategies(prev => {
        const next = new Set(prev)
        next.delete(deleteTarget.id)
        return next
      })
      if (expandedId === deleteTarget.id) setExpandedId(null)
      setDeleteTarget(null)
    } catch (err) {
      // Keep modal open on error so user can try again
    } finally {
      setDeleting(null)
    }
  }

  // Bulk selection handlers
  const toggleSelectMode = () => {
    setIsSelectMode(!isSelectMode)
    if (isSelectMode) {
      setSelectedStrategies(new Set())
    }
  }

  const toggleSelectStrategy = (id: string) => {
    setSelectedStrategies(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const selectAll = () => {
    setSelectedStrategies(new Set(strategies.map(s => s.id)))
  }

  const clearSelection = () => {
    setSelectedStrategies(new Set())
  }

  const handleBulkDeleteConfirm = async () => {
    if (selectedStrategies.size === 0) return

    try {
      setBulkDeleting(true)
      const deletePromises = Array.from(selectedStrategies).map(id =>
        fetch(`/api/strategies/${id}`, { method: 'DELETE' })
      )
      await Promise.all(deletePromises)

      setStrategies(prev => prev.filter(s => !selectedStrategies.has(s.id)))
      if (expandedId && selectedStrategies.has(expandedId)) setExpandedId(null)
      setSelectedStrategies(new Set())
      setShowBulkDeleteModal(false)
      setIsSelectMode(false)
    } catch (err) {
      // Keep modal open on error
    } finally {
      setBulkDeleting(false)
    }
  }

  const handleDeleteAllConfirm = async () => {
    if (strategies.length === 0) return

    try {
      setBulkDeleting(true)
      const deletePromises = strategies.map(s =>
        fetch(`/api/strategies/${s.id}`, { method: 'DELETE' })
      )
      await Promise.all(deletePromises)

      setStrategies([])
      setExpandedId(null)
      setSelectedStrategies(new Set())
      setShowDeleteAllModal(false)
      setIsSelectMode(false)
    } catch (err) {
      // Keep modal open on error
    } finally {
      setBulkDeleting(false)
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
        <div className="max-w-5xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Saved Strategies</h1>
            <p className="text-gray-400">
              View and manage your saved AI-generated strategies
            </p>
          </div>

          {/* Filter Tabs and Actions */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex gap-2">
              {(['all', 'backtest', 'forecast'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => setFilter(mode)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filter === mode
                      ? 'bg-[#4CC9F0] text-[#0D1B2A]'
                      : 'bg-[#14273F] text-gray-300 hover:bg-[#1a3352]'
                  }`}
                >
                  {mode === 'all' ? 'All' : mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>

            {/* Bulk Action Buttons */}
            {strategies.length > 0 && (
              <div className="flex items-center gap-2">
                {isSelectMode ? (
                  <>
                    <span className="text-sm text-gray-400">
                      {selectedStrategies.size} selected
                    </span>
                    <button
                      onClick={selectAll}
                      className="px-3 py-1.5 text-sm text-[#4CC9F0] hover:bg-[#4CC9F0]/10 rounded-lg transition-colors"
                    >
                      Select All
                    </button>
                    <button
                      onClick={clearSelection}
                      className="px-3 py-1.5 text-sm text-gray-400 hover:bg-gray-600/20 rounded-lg transition-colors"
                    >
                      Clear
                    </button>
                    {selectedStrategies.size > 0 && (
                      <button
                        onClick={() => setShowBulkDeleteModal(true)}
                        className="px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                      >
                        Delete Selected
                      </button>
                    )}
                    <button
                      onClick={toggleSelectMode}
                      className="p-1.5 text-gray-400 hover:text-white transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={toggleSelectMode}
                      className="px-3 py-1.5 text-sm text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
                    >
                      Select
                    </button>
                    <button
                      onClick={() => setShowDeleteAllModal(true)}
                      className="px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 border border-red-500/30 rounded-lg transition-colors"
                    >
                      Delete All
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Content */}
          <div className="bg-[#14273F] rounded-xl p-6">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-[#4CC9F0] animate-spin" />
              </div>
            ) : error ? (
              <div className="text-center py-8">
                <p className="text-red-400 mb-4">{error}</p>
                <button
                  onClick={fetchStrategies}
                  className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-4 py-2 rounded-full transition"
                >
                  Retry
                </button>
              </div>
            ) : strategies.length === 0 ? (
              <div className="text-center py-12">
                <Code className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400 mb-2">No saved strategies</p>
                <p className="text-gray-500 text-sm">
                  Generate a strategy using the AI popup and save it to see it here
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {strategies.map(strategy => (
                  <div
                    key={strategy.id}
                    className={`border rounded-lg overflow-hidden transition-colors ${
                      selectedStrategies.has(strategy.id)
                        ? 'border-[#4CC9F0]/50 bg-[#4CC9F0]/5'
                        : 'border-[#4CC9F0]/20'
                    }`}
                  >
                    {/* Strategy Header */}
                    <div
                      className="flex items-center justify-between p-4 cursor-pointer hover:bg-[#0D1B2A]/50 transition-colors"
                      onClick={() => {
                        if (isSelectMode) {
                          toggleSelectStrategy(strategy.id)
                        } else {
                          setExpandedId(expandedId === strategy.id ? null : strategy.id)
                        }
                      }}
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        {/* Checkbox in select mode */}
                        {isSelectMode && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleSelectStrategy(strategy.id)
                            }}
                            className="flex-shrink-0"
                          >
                            {selectedStrategies.has(strategy.id) ? (
                              <CheckSquare className="w-5 h-5 text-[#4CC9F0]" />
                            ) : (
                              <Square className="w-5 h-5 text-gray-500" />
                            )}
                          </button>
                        )}

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3">
                            <h3 className="font-medium text-white truncate">
                              {strategy.name}
                            </h3>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              strategy.mode === 'backtest'
                                ? 'bg-blue-500/20 text-blue-400'
                                : 'bg-green-500/20 text-green-400'
                            }`}>
                              {strategy.mode}
                            </span>
                          </div>
                          {strategy.description && (
                            <p className="text-sm text-gray-400 mt-1 truncate">
                              {strategy.description}
                            </p>
                          )}
                          <div className="flex items-center gap-1 text-xs text-gray-500 mt-2">
                            <Calendar className="w-3 h-3" />
                            {new Date(strategy.created_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>

                      {!isSelectMode && (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleEditClick(strategy)
                            }}
                            className="p-2 text-gray-400 hover:text-[#4CC9F0] transition-colors"
                            title="Edit strategy"
                          >
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteClick(strategy)
                            }}
                            disabled={deleting === strategy.id}
                            className="p-2 text-gray-400 hover:text-red-400 transition-colors disabled:opacity-50"
                            title="Delete strategy"
                          >
                            {deleting === strategy.id ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Expanded Code View */}
                    {expandedId === strategy.id && (
                      <div className="border-t border-[#4CC9F0]/20 bg-[#0D1B2A] p-4">
                        <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap font-mono">
                          {strategy.code}
                        </pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />

      {/* Single Delete Modal */}
      <DeleteConfirmModal
        isOpen={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDeleteConfirm}
        title="Delete Strategy"
        message="Are you sure you want to delete this strategy? This will permanently remove it from your saved strategies."
        itemName={deleteTarget?.name}
        loading={!!deleting}
      />

      {/* Bulk Delete Modal */}
      <DeleteConfirmModal
        isOpen={showBulkDeleteModal}
        onClose={() => setShowBulkDeleteModal(false)}
        onConfirm={handleBulkDeleteConfirm}
        title="Delete Selected Strategies"
        message={`Are you sure you want to delete ${selectedStrategies.size} selected ${selectedStrategies.size === 1 ? 'strategy' : 'strategies'}? This will permanently remove them from your saved strategies.`}
        loading={bulkDeleting}
      />

      {/* Delete All Modal */}
      <DeleteConfirmModal
        isOpen={showDeleteAllModal}
        onClose={() => setShowDeleteAllModal(false)}
        onConfirm={handleDeleteAllConfirm}
        title="Delete All Strategies"
        message={`Are you sure you want to delete all ${strategies.length} ${strategies.length === 1 ? 'strategy' : 'strategies'}? This will permanently remove them from your saved strategies.`}
        loading={bulkDeleting}
      />

      {/* Edit Strategy Modal */}
      <EditStrategyModal
        isOpen={!!editTarget}
        onClose={() => setEditTarget(null)}
        strategy={editTarget}
        onSave={handleEditSave}
      />
    </div>
  )
}
