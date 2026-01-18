"use client";

import { useState, useEffect, useRef } from 'react';
import { X, Folder, Loader2, Calendar, Code, Pencil } from 'lucide-react';
import EditStrategyModal from '@/components/ui/EditStrategyModal';
import type { HydratedStrategy } from '@/lib/types/strategy';

interface LoadStrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (strategy: HydratedStrategy) => void;
  mode: 'backtest' | 'forecast';
}

export default function LoadStrategyModal({
  isOpen,
  onClose,
  onSelect,
  mode,
}: LoadStrategyModalProps) {
  const [strategies, setStrategies] = useState<HydratedStrategy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editTarget, setEditTarget] = useState<HydratedStrategy | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Fetch strategies when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchStrategies();
    }
  }, [isOpen, mode]);

  const fetchStrategies = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`/api/strategies?mode=${mode}`);
      if (!res.ok) {
        throw new Error('Failed to fetch strategies');
      }
      const data = await res.json();
      setStrategies(data.strategies || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load strategies');
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = (strategy: HydratedStrategy) => {
    onSelect(strategy);
    onClose();
  };

  const handleEditSave = (updated: HydratedStrategy) => {
    setStrategies(prev => prev.map(s => s.id === updated.id ? updated : s));
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current) {
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
      onClick={handleOverlayClick}
      onKeyDown={handleKeyDown}
    >
      <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#4CC9F0]/20">
          <div className="flex items-center gap-3">
            <Folder className="w-5 h-5 text-[#4CC9F0]" />
            <div>
              <h3 className="font-semibold text-white">Load Saved Strategy</h3>
              <p className="text-sm text-gray-400">
                Select a {mode} strategy to load
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-[#4CC9F0] animate-spin" />
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <p className="text-red-400 mb-4">{error}</p>
              <button
                onClick={fetchStrategies}
                className="text-[#4CC9F0] hover:underline text-sm"
              >
                Try again
              </button>
            </div>
          ) : strategies.length === 0 ? (
            <div className="text-center py-12">
              <Code className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400 mb-2">No saved strategies</p>
              <p className="text-gray-500 text-sm">
                Generate a strategy and save it to see it here
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {strategies.map((strategy) => (
                <div
                  key={strategy.id}
                  className="border border-[#4CC9F0]/20 rounded-lg overflow-hidden hover:border-[#4CC9F0]/40 transition-colors"
                >
                  {/* Strategy Header */}
                  <div className="flex items-center justify-between p-4">
                    <div
                      className="flex-1 min-w-0 cursor-pointer"
                      onClick={() => setExpandedId(expandedId === strategy.id ? null : strategy.id)}
                    >
                      <h4 className="font-medium text-white truncate">
                        {strategy.name}
                      </h4>
                      {strategy.description && (
                        <p className="text-sm text-gray-400 truncate mt-0.5">
                          {strategy.description}
                        </p>
                      )}
                      <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(strategy.updated_at).toLocaleDateString()}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() => setEditTarget(strategy)}
                        className="p-2 text-gray-400 hover:text-[#4CC9F0] transition-colors"
                        title="View/Edit strategy"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleSelect(strategy)}
                        className="px-4 py-2 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-medium text-sm rounded-lg transition-colors"
                      >
                        Load
                      </button>
                    </div>
                  </div>

                  {/* Expanded Code Preview */}
                  {expandedId === strategy.id && (
                    <div className="border-t border-[#4CC9F0]/20 bg-[#0D1B2A] p-4">
                      <pre className="text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap font-mono max-h-48">
                        {strategy.code}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#4CC9F0]/20">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>

      {/* Edit Strategy Modal */}
      <EditStrategyModal
        isOpen={!!editTarget}
        onClose={() => setEditTarget(null)}
        strategy={editTarget}
        onSave={handleEditSave}
      />
    </div>
  );
}
