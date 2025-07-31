// Landing page with hero section and main navigation
"use client";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContexts";

export default function Home() {
  const { user, loading } = useAuth();
  return (
    <div className="min-h-screen bg-[#0D1B2A] text-white scroll-smooth">
      <Navbar />
      <main id="hero"></main>

      {/* Hero section with video background */}
      <section className="relative w-full min-h-[90vh] flex items-center justify-center bg-[#0D1B2A] overflow-hidden">
        {/* Curved overlay masks for visual effect */}
        <div className="pointer-events-none absolute -left-40 top-0 h-full w-[500px] bg-[#0D1B2A] rounded-r-full" />
        <div className="pointer-events-none absolute -right-40 top-0 h-full w-[500px] bg-[#0D1B2A] rounded-l-full" />

        {/* Background video loop */}
        <video
          className="absolute inset-0 w-full h-full object-cover opacity-20"
          autoPlay
          loop
          muted
          playsInline
        >
          <source src="/stockvid.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>

        {/* Main hero content and CTA buttons */}
        <div className="relative z-10 max-w-5xl px-6 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Forecast Smarter. Invest Better.
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-8">
            Visualize forecasts, configure algorithms, and build data-driven
            strategies — all in one dashboard.
          </p>
          <div className="flex justify-center gap-4">
            <Link
              href="/dashboard"
              className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-6 py-3 rounded-full transition"
            >
              Start Forecasting
            </Link>
            {!loading && !user && (
              <Link
                href="/auth/login"
                className="border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0] px-6 py-3 rounded-full transition"
              >
                Login to Save
              </Link>
            )}
          </div>
        </div>
      </section>

      {/* ─────────────────── How It Works ─────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20" id="how">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          {[
            {
              icon: "📊",
              title: "1 · Select Stocks",
              text: "Filter the Dow 30 by sector, market-cap, or technical criteria.",
            },
            {
              icon: "⚙️",
              title: "2 · Pick Algorithms",
              text: "Choose forecasting or machine-learning",
            },
            {
              icon: "🚀",
              title: "3 · Get Insights",
              text: "View forecasts, backtests, and recommended allocations instantly.",
            },
          ].map((card) => (
            <div
              key={card.title}
              className="bg-[#14273F] rounded-xl p-8 text-center shadow-lg flex flex-col items-center"
            >
              <div className="text-5xl mb-4">{card.icon}</div>
              <div className="font-semibold text-lg mb-2">{card.title}</div>
              <p className="text-gray-300">{card.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ─────────────────── Key Features ─────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20" id="features">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          Key Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          {[
            {
              title: "Real-Time Filtering",
              text: "Dynamic screener for Dow 30 tickers with instant preview.",
            },
            {
              title: "Multi-Model Forecasting",
              text: "Run ARIMA, Prophet, LSTM, or models hosted on our secure FastAPI-powered backend.",
            },
            {
              title: "Portfolio Simulation",
              text: "Visual equity curves, risk metrics, and allocation pie-charts.",
            },
          ].map((f) => (
            <div
              key={f.title}
              className="bg-[#14273F] rounded-xl p-8 shadow-lg"
            >
              <h3 className="font-semibold text-lg mb-3 text-[#4CC9F0]">
                {f.title}
              </h3>
              <p className="text-gray-300">{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ───────────── Sample Forecast Visualization ───────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20" id="preview">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          Live Preview
        </h2>
        <div className="relative w-full h-96 rounded-xl overflow-hidden shadow-lg">
          <video
          className="absolute inset-0 w-full h-full object-cover opacity-20"
          autoPlay
          loop
          muted
          playsInline
        >
          <source src="/stockvid.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        </div>
      </section>

      {/* ──────────────────── Call to Action ──────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20 flex flex-col items-center text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">
          Ready to elevate your research?
        </h2>
        <p className="text-lg text-gray-300 mb-10 max-w-2xl">
          Jump right in and start testing your strategies today.
        </p>
        <Link
          href={user ? "/dashboard" : "/auth/login"}
          className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-8 py-4 rounded-full transition"
        >
          Get Started
        </Link>
      </section>

      {/* ─────────────────────── Footer ─────────────────────── */}
      <Footer />
    </div>
  );
}