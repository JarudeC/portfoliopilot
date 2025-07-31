// Login page with form and authentication handling
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContexts";
import AuthLayout from "@/components/auth/AuthLayout";
import SignInForm from "@/components/auth/SignInForm";

export default function LoginPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  const handleToggleMode = () => {
    router.push("/auth/signup");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0D1B2A] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-[#4CC9F0]/30 border-t-[#4CC9F0] rounded-full animate-spin" />
      </div>
    );
  }

  if (user) {
    return null; // Will redirect via useEffect
  }

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Sign in to access your portfolio dashboard"
    >
      <SignInForm onToggleMode={handleToggleMode} />
    </AuthLayout>
  );
}