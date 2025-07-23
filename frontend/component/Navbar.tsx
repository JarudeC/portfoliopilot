"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const navLink = "text-sm font-medium text-white hover:text-[#4CC9F0] transition-colors";
  const pill = "px-5 py-2 text-sm font-medium rounded-full transition-colors";

  return (
    <nav className="fixed inset-x-0 top-0 z-50 bg-[#0D1B2A] shadow-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 md:px-8 lg:px-32">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <Image
            src="/icon.png"
            alt="PortfolioPilot Logo"
            width={32}
            height={32}
            className="rounded-full"
          />
          <span className="text-xl font-bold text-white">PortfolioPilot</span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center space-x-6">
          <Link href="/dashboard" className={navLink}>
            Dashboard
          </Link>
          <Link href="/history" className={navLink}>
            History
          </Link>
          <Link
            href="/api/auth/signin"
            className={`${pill} border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0]`}
          >
            Sign In
          </Link>
          <Link
            href="/api/auth/login"
            className={`${pill} bg-[#4CC9F0] text-[#0D1B2A] hover:bg-[#3A86FF]`}
          >
            Login
          </Link>
        </div>

        {/* Mobile toggle */}
        <button
          onClick={() => setOpen((p) => !p)}
          className="md:hidden p-2 text-white focus:outline-none focus:ring-2 focus:ring-[#4CC9F0]"
          aria-label="Toggle menu"
        >
          <svg
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            {open ? (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            ) : (
              <>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 12h16" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 18h16" />
              </>
            )}
          </svg>
        </button>
      </div>

      {/* Mobile drawer */}
      {open && (
        <div className="md:hidden flex flex-col gap-4 px-4 pb-6">
          <Link href="/dashboard" className={navLink} onClick={() => setOpen(false)}>
            Dashboard
          </Link>
          <Link href="/history" className={navLink} onClick={() => setOpen(false)}>
            History
          </Link>
          <Link
            href="/api/auth/signin"
            className={`${pill} border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0]`}
            onClick={() => setOpen(false)}
          >
            Sign In
          </Link>
          <Link
            href="/api/auth/login"
            className={`${pill} bg-[#4CC9F0] text-[#0D1B2A] hover:bg-[#3A86FF]`}
            onClick={() => setOpen(false)}
          >
            Login
          </Link>
        </div>
      )}
    </nav>
  );
}