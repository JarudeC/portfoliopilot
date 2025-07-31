"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContexts";
import { FaGoogle } from "react-icons/fa";

interface OAuthButtonsProps {
  onError?: (error: string) => void;
}

export default function OAuthButtons({ onError }: OAuthButtonsProps) {
  const { signInWithProvider } = useAuth();
  const [loading, setLoading] = useState(false);

  const handleGoogleSignIn = async () => {
    try {
      setLoading(true);
      await signInWithProvider('google');
    } catch (error: any) {
      console.error('Google sign in error:', error);
      onError?.(error.message || 'Failed to sign in with Google');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <button
        onClick={handleGoogleSignIn}
        disabled={loading}
        className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white hover:bg-gray-50 text-gray-900 font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-900 rounded-full animate-spin" />
        ) : (
          <FaGoogle className="w-5 h-5 text-red-500" />
        )}
        Continue with Google
      </button>

      <div className="relative my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-[#1B263B]" />
        </div>
        <div className="relative flex justify-center text-xs">
          <span className="px-3 bg-[#14273F] text-gray-400">Or continue with email</span>
        </div>
      </div>
    </div>
  );
}