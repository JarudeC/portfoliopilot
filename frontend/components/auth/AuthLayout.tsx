"use client";

import Image from "next/image";

interface AuthLayoutProps {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}

export default function AuthLayout({ title, subtitle, children }: AuthLayoutProps) {
  return (
    <div className="min-h-screen bg-[#0D1B2A] flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo and branding */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Image
              src="/icon.png"
              alt="PortfolioPilot Logo"
              width={40}
              height={40}
              className="rounded-full"
            />
            <span className="text-2xl font-bold text-white">PortfolioPilot</span>
          </div>
          <h1 className="text-xl font-semibold text-white mb-2">{title}</h1>
          <p className="text-sm text-gray-400">{subtitle}</p>
        </div>

        {/* Auth form container */}
        <div className="bg-[#14273F] rounded-xl p-6 border border-[#1B263B]">
          {children}
        </div>

        {/* Footer text */}
        <p className="text-center text-xs text-gray-500 mt-6">
          By continuing, you agree to our Terms of Service and Privacy Policy
        </p>
      </div>
    </div>
  );
}