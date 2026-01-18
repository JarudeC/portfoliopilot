"use client";

import { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { X, Save, Loader2 } from 'lucide-react';
import type { HydratedStrategy } from '@/lib/types/strategy';

interface EditStrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  strategy: HydratedStrategy | null;
  onSave: (updated: HydratedStrategy) => void;
}

export default function EditStrategyModal({
  isOpen,
  onClose,
  strategy,
  onSave,
}: EditStrategyModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [code, setCode] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset form when strategy changes
  useEffect(() => {
    if (strategy && isOpen) {
      setName(strategy.name);
      setDescription(strategy.description || '');
      setCode(strategy.code);
      setError(null);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [strategy, isOpen]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const handleSave = async () => {
    if (!strategy) return;

    if (!name.trim()) {
      setError('Please enter a strategy name');
      return;
    }

    if (!code.trim()) {
      setError('Strategy code cannot be empty');
      return;
    }

    try {
      setSaving(true);
      setError(null);

      const res = await fetch(`/api/strategies/${strategy.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim(),
          code: code,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || 'Failed to update strategy');
      }

      const data = await res.json();
      onSave(data.strategy);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategy');
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
      handleSave();
    }
  };

  if (!isOpen || !strategy) return null;

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={handleOverlayClick}
      onKeyDown={handleKeyDown}
    >
      <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#4CC9F0]/20">
          <h3 className="font-semibold text-white">Edit Strategy</h3>
          <button
            onClick={onClose}
            disabled={saving}
            className="text-gray-400 hover:text-white transition-colors p-1 disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Name */}
          <div>
            <label htmlFor="edit-strategy-name" className="block text-sm font-medium text-gray-300 mb-1">
              Strategy Name *
            </label>
            <input
              ref={inputRef}
              id="edit-strategy-name"
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
            <label htmlFor="edit-strategy-description" className="block text-sm font-medium text-gray-300 mb-1">
              Description
            </label>
            <textarea
              id="edit-strategy-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={saving}
              placeholder="Optional: describe what this strategy does"
              className="w-full px-3 py-2 bg-[#0D1B2A] border border-[#4CC9F0]/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors resize-none disabled:opacity-50"
              rows={2}
              maxLength={500}
            />
          </div>

          {/* Code Editor */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Strategy Code *
            </label>
            <div className="border border-[#4CC9F0]/20 rounded-lg overflow-hidden">
              <Editor
                height="300px"
                defaultLanguage="typescript"
                value={code}
                onChange={(value) => setCode(value || '')}
                theme="vs-dark"
                options={{
                  readOnly: saving,
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  wordWrap: 'on',
                  automaticLayout: true,
                }}
              />
            </div>
          </div>

          {/* Mode indicator */}
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>Mode:</span>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              strategy.mode === 'backtest'
                ? 'bg-blue-500/20 text-blue-400'
                : 'bg-green-500/20 text-green-400'
            }`}>
              {strategy.mode}
            </span>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-[#4CC9F0]/20">
          <p className="text-xs text-gray-500">
            <kbd className="px-1 py-0.5 bg-gray-700 rounded">Ctrl+Enter</kbd> to save
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              disabled={saving}
              className="px-4 py-2 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || !name.trim() || !code.trim()}
              className="flex items-center gap-2 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-medium px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save Changes
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
