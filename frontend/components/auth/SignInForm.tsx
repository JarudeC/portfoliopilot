"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContexts";
import { useRouter } from "next/navigation";
import Link from "next/link";
import OAuthButtons from "./OAuthButtons";

interface SignInFormProps {
  onToggleMode?: () => void;
}

export default function SignInForm({ onToggleMode }: SignInFormProps) {
  const { signIn } = useAuth();
  const router = useRouter();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.email || !formData.password) {
      setError("Please fill in all fields");
      return;
    }

    try {
      setLoading(true);
      setError("");
      await signIn(formData.email, formData.password);
      router.push("/");
    } catch (error: any) {
      console.error("Sign in error:", error);
      setError(error.message || "Failed to sign in");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  return (
    <div>
      <OAuthButtons onError={setError} />
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
            Email address
          </label>
          <input
            id="email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            required
            className="w-full px-4 py-3 bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors"
            placeholder="Enter your email"
          />
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            required
            className="w-full px-4 py-3 bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors"
            placeholder="Enter your password"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <div className="flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-[#0D1B2A]/30 border-t-[#0D1B2A] rounded-full animate-spin" />
              Signing in...
            </div>
          ) : (
            "Sign In"
          )}
        </button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-gray-400">
          Don't have an account?{" "}
          {onToggleMode ? (
            <button
              onClick={onToggleMode}
              className="text-[#4CC9F0] hover:text-[#3A86FF] font-medium transition-colors"
            >
              Sign up
            </button>
          ) : (
            <Link
              href="/auth/signup"
              className="text-[#4CC9F0] hover:text-[#3A86FF] font-medium transition-colors"
            >
              Sign up
            </Link>
          )}
        </p>
      </div>
    </div>
  );
}