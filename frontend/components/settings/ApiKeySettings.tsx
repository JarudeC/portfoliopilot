/**
 * Component for managing Anthropic API key settings.
 *
 * Features:
 * - Input field for entering new API key
 * - Shows masked preview when key exists (e.g., "sk-ant-...xxxx")
 * - Validates key format before saving
 * - Delete functionality to remove saved key
 * - Loading states and error handling
 */

'use client';

import { useState, useEffect } from 'react';
import DeleteConfirmModal from '@/components/ui/DeleteConfirmModal';

interface KeyInfo {
  hasKey: boolean;
  keyPreview?: string;
  updatedAt?: string;
}

export default function ApiKeySettings() {
  const [keyInfo, setKeyInfo] = useState<KeyInfo | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Fetch current key status on mount
  useEffect(() => {
    fetchKeyInfo();
  }, []);

  const fetchKeyInfo = async () => {
    try {
      setLoading(true);
      setError(null);

      const res = await fetch('/api/settings/api-key?provider=anthropic');
      if (!res.ok) {
        throw new Error('Failed to fetch API key status');
      }

      const data = await res.json();
      setKeyInfo(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load API key status');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveKey = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      setSuccess(null);

      const res = await fetch('/api/settings/api-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'anthropic',
          apiKey: apiKey.trim(),
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Failed to save API key');
      }

      setSuccess('API key saved successfully');
      setApiKey('');
      setKeyInfo({
        hasKey: true,
        keyPreview: data.keyPreview,
      });

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save API key');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteKey = async () => {
    try {
      setDeleting(true);
      setError(null);
      setSuccess(null);

      const res = await fetch('/api/settings/api-key?provider=anthropic', {
        method: 'DELETE',
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to delete API key');
      }

      setSuccess('API key deleted successfully');
      setKeyInfo({ hasKey: false });
      setShowDeleteModal(false);

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete API key');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-[#14273F] rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">API Keys</h2>
        <div className="text-gray-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="bg-[#14273F] rounded-xl p-6">
      <h2 className="text-xl font-semibold mb-2">API Keys</h2>
      <p className="text-gray-400 text-sm mb-6">
        Add your own Anthropic API key to use Claude AI features. Your key is encrypted and stored securely.
      </p>

      {/* Anthropic Claude Section */}
      <div className="border border-[#1B3A57] rounded-lg p-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-[#F97316] rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">C</span>
          </div>
          <div>
            <h3 className="font-medium">Anthropic Claude</h3>
            <p className="text-gray-400 text-sm">Used for AI-powered strategy generation</p>
          </div>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-2 rounded-lg mb-4">
            {error}
          </div>
        )}
        {success && (
          <div className="bg-green-500/10 border border-green-500/30 text-green-400 px-4 py-2 rounded-lg mb-4">
            {success}
          </div>
        )}

        {/* Current Key Status */}
        {keyInfo?.hasKey && (
          <div className="flex items-center justify-between bg-[#0D1B2A] rounded-lg px-4 py-3 mb-4">
            <div>
              <p className="text-sm text-gray-400">Current key</p>
              <p className="font-mono text-sm">{keyInfo.keyPreview}</p>
            </div>
            <button
              onClick={() => setShowDeleteModal(true)}
              disabled={deleting}
              className="text-red-400 hover:text-red-300 text-sm px-3 py-1 rounded border border-red-400/30 hover:border-red-400/50 transition disabled:opacity-50"
            >
              Delete
            </button>
          </div>
        )}

        {/* Add/Update Key Form */}
        <form onSubmit={handleSaveKey}>
          <label className="block text-sm text-gray-400 mb-2">
            {keyInfo?.hasKey ? 'Replace with new key' : 'Enter your API key'}
          </label>
          <div className="flex gap-2">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-..."
              className="flex-1 bg-[#0D1B2A] border border-[#1B3A57] rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] transition"
            />
            <button
              type="submit"
              disabled={saving || !apiKey.trim()}
              className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-6 py-2 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
          <p className="text-gray-500 text-xs mt-2">
            Get your API key from{' '}
            <a
              href="https://console.anthropic.com/settings/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#4CC9F0] hover:underline"
            >
              console.anthropic.com
            </a>
          </p>
        </form>
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleDeleteKey}
        title="Delete API Key"
        message="Are you sure you want to delete your API key? You will need to add a new key to use AI features."
        loading={deleting}
      />
    </div>
  );
}
