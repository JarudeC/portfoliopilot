/**
 * Settings page for user preferences and API key management.
 *
 * Currently includes:
 * - API Key settings (Anthropic Claude)
 *
 * Future sections can be added for:
 * - Notification preferences
 * - Display settings
 * - Data export/import
 */

'use client';

import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContexts';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import ApiKeySettings from '@/components/settings/ApiKeySettings';

export default function SettingsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  // Redirect to home if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/');
    }
  }, [user, authLoading, router]);

  // Show loading state while checking auth
  if (authLoading || !user) {
    return (
      <div className="min-h-screen bg-[#0D1B2A] flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0D1B2A] text-white flex flex-col">
      <Navbar />

      <main className="flex-1 pt-[72px] pb-20 px-4 lg:px-16">
        <div className="max-w-3xl mx-auto">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Settings</h1>
            <p className="text-gray-400">
              Manage your account settings and preferences
            </p>
          </div>

          {/* Settings Sections */}
          <div className="space-y-6">
            {/* API Keys Section */}
            <ApiKeySettings />

            {/* Additional settings sections can be added here */}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
