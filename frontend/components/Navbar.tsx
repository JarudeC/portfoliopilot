// Navigation bar with user authentication and mobile menu
"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContexts";

export default function Navbar() {
  const { user, loading, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  const navLink = "text-sm font-medium text-white hover:text-[#4CC9F0] transition-colors";
  const pill = "px-5 py-2 text-sm font-medium rounded-full transition-colors";

  // Handle click outside to close user menu
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);


  const handleSignOut = async () => {
    try {
      await signOut();
      setUserMenuOpen(false);
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  const getUserDisplayName = () => {
    if (!user) return '';
    return user.user_metadata?.full_name || 
           user.user_metadata?.name || 
           user.email?.split('@')[0] || 
           'User';
  };

  const getUserInitials = () => {
    const displayName = getUserDisplayName();
    return displayName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  };

  return (
    <nav className="fixed inset-x-0 top-0 z-50 bg-[#0D1B2A] shadow-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 md:px-8 lg:px-32">
        {/* Brand logo and name */}
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

        {/* Desktop navigation menu */}
        <div className="hidden md:flex items-center space-x-6">
          <Link href="/dashboard" className={navLink}>
            Dashboard
          </Link>
          
          {user && (
            <Link href="/history" className={navLink}>
              History
            </Link>
          )}

          {loading ? (
            <div className="w-6 h-6 border-2 border-[#4CC9F0]/30 border-t-[#4CC9F0] rounded-full animate-spin" />
          ) : user ? (
            // User dropdown menu
            <div className="relative" ref={userMenuRef}>
              <button
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center gap-2 text-white hover:text-[#4CC9F0] transition-colors"
              >
                <div className="w-8 h-8 bg-[#4CC9F0] text-[#0D1B2A] rounded-full flex items-center justify-center text-sm font-semibold">
                  {getUserInitials()}
                </div>
                <span className="text-sm font-medium">{getUserDisplayName()}</span>
                <svg
                  className={`w-4 h-4 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {userMenuOpen && (
                <div className="absolute right-0 mt-2 min-w-48 w-max max-w-80 bg-[#14273F] border border-[#1B263B] rounded-lg shadow-lg py-2">
                  <div className="px-4 py-2 border-b border-[#1B263B]">
                    <p className="text-sm font-medium text-white truncate">{getUserDisplayName()}</p>
                    <p className="text-xs text-gray-400 truncate">{user.email}</p>
                  </div>
                  <button
                    onClick={handleSignOut}
                    className="w-full text-left px-4 py-2 text-sm text-white hover:bg-[#1F2E45] transition-colors"
                  >
                    Sign out
                  </button>
                </div>
              )}
            </div>
          ) : (
            // Login and signup buttons
            <>
              <Link
                href="/auth/login"
                className={`${pill} border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0]`}
              >
                Login
              </Link>
              <Link
                href="/auth/signup"
                className={`${pill} bg-[#4CC9F0] text-[#0D1B2A] hover:bg-[#3A86FF]`}
              >
                Sign Up
              </Link>
            </>
          )}
        </div>

        {/* Hamburger menu button */}
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

      {/* Mobile navigation drawer */}
      {open && (
        <div className="md:hidden flex flex-col gap-4 px-4 pb-6">
          <Link href="/dashboard" className={navLink} onClick={() => setOpen(false)}>
            Dashboard
          </Link>
          
          {user && (
            <Link href="/history" className={navLink} onClick={() => setOpen(false)}>
              History
            </Link>
          )}

          {loading ? (
            <div className="flex justify-center py-2">
              <div className="w-6 h-6 border-2 border-[#4CC9F0]/30 border-t-[#4CC9F0] rounded-full animate-spin" />
            </div>
          ) : user ? (
            // Mobile user profile section
            <div className="space-y-3 pt-2 border-t border-[#1B263B]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#4CC9F0] text-[#0D1B2A] rounded-full flex items-center justify-center text-sm font-semibold">
                  {getUserInitials()}
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{getUserDisplayName()}</p>
                  <p className="text-xs text-gray-400">{user.email}</p>
                </div>
              </div>
              <button
                onClick={() => {
                  handleSignOut();
                  setOpen(false);
                }}
                className="w-full text-left px-3 py-2 text-sm text-white hover:text-[#4CC9F0] transition-colors"
              >
                Sign out
              </button>
            </div>
          ) : (
            // Login and signup buttons
            <>
              <Link
                href="/auth/login"
                onClick={() => setOpen(false)}
                className={`${pill} border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0]`}
              >
                Login
              </Link>
              <Link
                href="/auth/signup"
                onClick={() => setOpen(false)}
                className={`${pill} bg-[#4CC9F0] text-[#0D1B2A] hover:bg-[#3A86FF]`}
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      )}
    </nav>
  );
}