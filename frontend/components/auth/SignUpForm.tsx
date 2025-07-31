"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContexts";
import Link from "next/link";
import OAuthButtons from "./OAuthButtons";

interface SignUpFormProps {
  onToggleMode?: () => void;
}

export default function SignUpForm({ onToggleMode }: SignUpFormProps) {
  const { signUp } = useAuth();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.email || !formData.password || !formData.confirmPassword) {
      setError("Please fill in all fields");
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long");
      return;
    }

    try {
      setLoading(true);
      setError("");
      await signUp(formData.email, formData.password);
      setSuccess(true);
    } catch (error: any) {
      console.error("Sign up error:", error);
      setError(error.message || "Failed to create account");
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

  if (success) {
    return (
      <div className="text-center space-y-4">
        <div className="w-16 h-16 bg-green-500/10 rounded-full flex items-center justify-center mx-auto">
          <svg className="w-8 h-8 text-green-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-white">Check your email</h3>
        <p className="text-sm text-gray-400">
          We've sent you a confirmation link at <strong>{formData.email}</strong>. 
          Click the link to verify your account and complete registration.
        </p>
        <Link
          href="/auth/login"
          className="w-full bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold py-3 rounded-lg transition-colors inline-block text-center"
        >
          Go to Login
        </Link>
      </div>
    );
  }

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
            placeholder="Create a password (min. 6 characters)"
          />
        </div>

        <div>
          <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-2">
            Confirm password
          </label>
          <input
            id="confirmPassword"
            name="confirmPassword"
            type="password"
            value={formData.confirmPassword}
            onChange={handleChange}
            required
            className="w-full px-4 py-3 bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors"
            placeholder="Confirm your password"
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
              Creating account...
            </div>
          ) : (
            "Create Account"
          )}
        </button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-gray-400">
          Already have an account?{" "}
          {onToggleMode ? (
            <button
              onClick={onToggleMode}
              className="text-[#4CC9F0] hover:text-[#3A86FF] font-medium transition-colors"
            >
              Log in
            </button>
          ) : (
            <Link
              href="/auth/login"
              className="text-[#4CC9F0] hover:text-[#3A86FF] font-medium transition-colors"
            >
              Sign in
            </Link>
          )}
        </p>
      </div>
    </div>
  );
}