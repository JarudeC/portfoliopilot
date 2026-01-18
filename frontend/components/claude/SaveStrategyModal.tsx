"use client";

import { useState, useRef, useEffect } from 'react';
import { X, Save, Loader2 } from 'lucide-react';

interface SaveStrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description: string) => Promise<void>;
  onUpdate?: (name: string, description: string) => Promise<void>;
  defaultDescription?: string;
  existingStrategyName?: string;
  loading?: boolean;
}

export default function SaveStrategyModal({
  isOpen,
  onClose,
  onSave,
  onUpdate,
  defaultDescription = '',
  existingStrategyName,
  loading = false,
}: SaveStrategyModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState(defaultDescription);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setName(existingStrategyName || '');
      setDescription(defaultDescription);
      setError(null);
    }
  }, [isOpen, defaultDescription, existingStrategyName]);

  const handleSaveNew = async () => {
    if (!name.trim()) {
      setError('Please enter a strategy name');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      await onSave(name.trim(), description.trim());
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategy');
    } finally {
      setSaving(false);
    }
  };

  const handleUpdate = async () => {
    if (!onUpdate) return;

    if (!name.trim()) {
      setError('Please enter a strategy name');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      await onUpdate(name.trim(), description.trim());
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update strategy');
    } finally {
      setSaving(false);
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current && !saving) {
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !saving) {
      onClose();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSaveNew();
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
      <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl max-w-md w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#4CC9F0]/20">
          <h3 className="font-semibold text-white">Save Strategy</h3>
          <button
            onClick={onClose}
            disabled={saving}
            className="text-gray-400 hover:text-white transition-colors p-1 disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Strategy Name */}
          <div>
            <label htmlFor="strategy-name" className="block text-sm font-medium text-gray-300 mb-1">
              Strategy Name *
            </label>
            <input
              ref={inputRef}
              id="strategy-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={saving}
              placeholder="e.g., Momentum + Low Vol"
              className="w-full px-3 py-2 bg-[#0D1B2A] border border-[#4CC9F0]/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors disabled:opacity-50"
              maxLength={100}
            />
          </div>

          {/* Description */}
          <div>
            <label htmlFor="strategy-description" className="block text-sm font-medium text-gray-300 mb-1">
              Description
            </label>
            <textarea
              id="strategy-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={saving}
              placeholder="Optional: describe what this strategy does"
              className="w-full px-3 py-2 bg-[#0D1B2A] border border-[#4CC9F0]/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors resize-none disabled:opacity-50"
              rows={3}
              maxLength={500}
            />
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2">
            {existingStrategyName && onUpdate && (
              <button
                onClick={handleUpdate}
                disabled={saving || loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 border border-[#4CC9F0] text-[#4CC9F0] hover:bg-[#4CC9F0]/10 rounded-lg transition-colors disabled:opacity-50"
              >
                {saving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                Update "{existingStrategyName}"
              </button>
            )}

            <button
              onClick={handleSaveNew}
              disabled={saving || loading || !name.trim()}
              className="flex-1 flex items-center justify-center gap-2 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-medium px-4 py-2.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {existingStrategyName ? 'Save as New' : 'Save Strategy'}
            </button>
          </div>

          {/* Keyboard hint */}
          <p className="text-xs text-gray-500 text-center">
            <kbd className="px-1 py-0.5 bg-gray-700 rounded">Ctrl+Enter</kbd> to save
          </p>
        </div>
      </div>
    </div>
  );
}
