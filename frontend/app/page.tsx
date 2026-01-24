// Landing page with hero section and main navigation
"use client";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContexts";
import {
  ChartBar,
  Bot,
  Rocket,
  Target,
  Zap,
  Lock,
  TrendingUp,
  Database,
  Settings,
  Brain,
  BarChart3,
  Gauge,
  Palette,
  RotateCcw,
  Bookmark
} from "lucide-react";

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
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-[#4CC9F0] via-[#3A86FF] to-[#7209B7] bg-clip-text text-transparent">
            AI-Enhanced Portfolio Construction
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-8 max-w-4xl">
            Build sophisticated investment portfolios using machine learning, classical algorithms, and custom AI-generated strategies powered by Anthropic Claude.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              href="/dashboard"
              className="bg-gradient-to-r from-[#4CC9F0] to-[#3A86FF] hover:from-[#3A86FF] hover:to-[#7209B7] text-[#0D1B2A] font-semibold px-8 py-4 rounded-full transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Start Building Portfolios
            </Link>
            {!loading && !user && (
              <Link
                href="/auth/login"
                className="border-2 border-[#4CC9F0] text-white hover:bg-[#4CC9F0] hover:text-[#0D1B2A] px-8 py-4 rounded-full transition-all duration-300 font-semibold"
              >
                Sign In to Save Strategies & History
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
              icon: <ChartBar className="w-12 h-12 text-[#4CC9F0]" />,
              title: "1 · Select Stocks",
              text: "Choose up to 8 stocks from the DOW30 to build your portfolio.",
            },
            {
              icon: <Bot className="w-12 h-12 text-[#4CC9F0]" />,
              title: "2 · Choose Strategy",
              text: "Select pre-built algorithms or create custom AI-generated strategies using natural language.",
            },
            {
              icon: <Rocket className="w-12 h-12 text-[#4CC9F0]" />,
              title: "3 · Analyze Results",
              text: "View forecasts, backtests, portfolio weights, and comprehensive performance metrics.",
            },
          ].map((card) => (
            <div
              key={card.title}
              className="bg-[#14273F] rounded-xl p-8 text-center shadow-lg flex flex-col items-center"
            >
              <div className="mb-4">{card.icon}</div>
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {[
            {
              icon: <Target className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Stock Selection",
              text: "Choose from DOW30 stocks with historical price data from Yahoo Finance.",
            },
            {
              icon: <Brain className="w-8 h-8 text-[#4CC9F0]" />,
              title: "AI Strategy Generation",
              text: "Describe your investment approach in natural language and let Claude generate custom algorithms.",
            },
            {
              icon: <TrendingUp className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Pre-built Algorithms",
              text: "Access proven models: ARIMA, LSTM, Autoformer, Markowitz optimization, and reinforcement learning.",
            },
            {
              icon: <Zap className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Real-time Backtesting",
              text: "Test strategies against historical data with comprehensive performance metrics and risk analysis.",
            },
            {
              icon: <BarChart3 className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Interactive Visualizations",
              text: "View portfolio compositions, equity curves, forecast charts, and performance analytics.",
            },
            {
              icon: <Database className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Session History",
              text: "View training history, compare performance across runs, and analyze past experiments.",
            },
            {
              icon: <Bookmark className="w-8 h-8 text-[#4CC9F0]" />,
              title: "Custom Strategy Library",
              text: "Save your Claude-generated strategies and reuse them across sessions without re-prompting.",
            },
          ].map((f) => (
            <div
              key={f.title}
              className="bg-[#14273F] rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 border border-[#4CC9F0]/10 hover:border-[#4CC9F0]/30"
            >
              <div className="mb-4">{f.icon}</div>
              <h3 className="font-semibold text-lg mb-3 text-[#4CC9F0]">
                {f.title}
              </h3>
              <p className="text-gray-300 text-sm leading-relaxed">{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ─────────────────── Available Algorithms ─────────────────── */}
      <section className="bg-[#14273F]/50 py-20">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-center">
            Powerful Algorithm Suite
          </h2>
          <p className="text-lg text-gray-300 mb-12 text-center max-w-3xl mx-auto">
            Choose from proven classical algorithms or create custom strategies using natural language descriptions
          </p>

          {/* Pre-built Algorithms - Full Width */}
          <div className="bg-[#0D1B2A] rounded-xl p-8 border border-[#4CC9F0]/20 mb-8">
            <h3 className="text-2xl font-bold mb-6 text-[#4CC9F0] flex items-center gap-3">
              <Settings className="w-8 h-8" />
              Pre-built Algorithms
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Forecasting Models */}
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">Forecasting Models</div>
                {[
                  { name: "ARIMA", desc: "Classical statistical model that captures auto-regressive patterns and moving averages to predict future prices" },
                  { name: "LSTM", desc: "Deep learning neural network designed to learn long-term dependencies in sequential price data" },
                  { name: "Autoformer", desc: "State-of-the-art transformer with auto-correlation mechanism for long-horizon time series forecasting" },
                ].map((algo, index) => (
                  <div key={index} className="border-l-4 border-[#4CC9F0] pl-4 py-2 mb-2">
                    <div className="font-semibold text-white">{algo.name}</div>
                    <div className="text-sm text-gray-400">{algo.desc}</div>
                  </div>
                ))}
              </div>
              {/* Portfolio Optimization */}
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">Portfolio Optimization</div>
                {[
                  { name: "Naive Markowitz", desc: "Classic mean-variance optimization balancing expected returns against portfolio risk using covariance estimation" },
                  { name: "GMVP", desc: "Global minimum variance portfolio that minimizes overall risk through covariance clustering and regularization" },
                  { name: "PPN", desc: "Policy Portfolio Network - deep reinforcement learning that learns allocation policies directly from price data" },
                  { name: "Margin Trader", desc: "A2C reinforcement learning agent supporting both long and short positions with configurable leverage" },
                ].map((algo, index) => (
                  <div key={index} className="border-l-4 border-[#4CC9F0] pl-4 py-2 mb-2">
                    <div className="font-semibold text-white">{algo.name}</div>
                    <div className="text-sm text-gray-400">{algo.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* AI-Generated Strategies - Full Width */}
          <div className="bg-gradient-to-br from-[#4CC9F0]/10 to-[#7209B7]/10 rounded-xl p-8 border border-[#4CC9F0]/30">
            <h3 className="text-2xl font-bold mb-6 text-[#4CC9F0] flex items-center gap-3">
              <Bot className="w-8 h-8" />
              AI-Generated Strategies
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Example Input */}
              <div className="bg-[#0D1B2A]/80 rounded-lg p-5 border border-[#4CC9F0]/20">
                <div className="font-semibold text-white mb-3">Natural Language Input</div>
                <div className="text-sm text-gray-300 italic mb-4">
                  "Create a momentum strategy that allocates higher weights to stocks with strong 30-day performance but limits individual positions to 25%"
                </div>
                <div className="text-xs text-[#4CC9F0]">→ Generates TypeScript algorithm automatically</div>
              </div>
              {/* Features */}
              <div className="flex flex-col justify-center space-y-3">
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 bg-[#4CC9F0] rounded-full"></span>
                  <span className="text-gray-300">Security validation & code analysis</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 bg-[#4CC9F0] rounded-full"></span>
                  <span className="text-gray-300">Real-time compilation & execution</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 bg-[#4CC9F0] rounded-full"></span>
                  <span className="text-gray-300">Fallback strategies for reliability</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 bg-[#4CC9F0] rounded-full"></span>
                  <span className="text-gray-300">Preview & edit generated code</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ───────────── Sample Forecast Visualization ───────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20" id="preview">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          Demo
        </h2>
        <div className="relative w-full max-w-6xl mx-auto rounded-xl overflow-hidden shadow-2xl bg-gradient-to-br from-[#4CC9F0] via-[#3A86FF] to-[#7209B7] p-1">
          <div className="bg-black rounded-lg overflow-hidden">
            <video
              className="w-full h-auto"
              autoPlay
              loop
              muted
              playsInline
              controls
            >
              <source src="/PortfolioPilot_Demo.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      </section>

      {/* ─────────────────── Technical Stack & Benefits ─────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          {/* Technical Stack */}
          <div>
            <h2 className="text-3xl font-bold mb-8 text-[#4CC9F0]">Built with Modern Tech</h2>
            <div className="grid grid-cols-2 gap-6">
              {[
                { category: "Frontend", techs: ["Next.js 15", "TypeScript", "React", "Tailwind CSS"] },
                { category: "Backend", techs: ["FastAPI", "Python", "PyTorch", "Scikit-learn"] },
                { category: "Database", techs: ["Supabase", "PostgreSQL", "JWT Auth", "Real-time"] },
                { category: "AI/ML", techs: ["Anthropic Claude", "LSTM", "Transformers", "RL"] }
              ].map((stack, index) => (
                <div key={index} className="bg-[#14273F] rounded-lg p-4 border border-[#4CC9F0]/20">
                  <h4 className="font-semibold text-white mb-3">{stack.category}</h4>
                  <ul className="space-y-1">
                    {stack.techs.map((tech, i) => (
                      <li key={i} className="text-sm text-gray-300 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 bg-[#4CC9F0] rounded-full"></span>
                        {tech}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Benefits */}
          <div>
            <h2 className="text-3xl font-bold mb-8 text-[#4CC9F0]">Why Choose Portfolio Pilot?</h2>
            <div className="space-y-4">
              {[
                { icon: <Target className="w-6 h-6 text-[#4CC9F0]" />, title: "Precision", desc: "Advanced algorithms with rigorous backtesting and validation" },
                { icon: <Zap className="w-6 h-6 text-[#4CC9F0]" />, title: "Speed", desc: "Real-time strategy generation and portfolio optimization" },
                { icon: <Lock className="w-6 h-6 text-[#4CC9F0]" />, title: "Security", desc: "Multi-layer security validation for AI-generated code" },
                { icon: <Gauge className="w-6 h-6 text-[#4CC9F0]" />, title: "Performance", desc: "Comprehensive metrics and risk analysis tools" },
                { icon: <Palette className="w-6 h-6 text-[#4CC9F0]" />, title: "Usability", desc: "Intuitive interface with natural language strategy creation" },
                { icon: <RotateCcw className="w-6 h-6 text-[#4CC9F0]" />, title: "Reliability", desc: "Robust fallback systems ensure consistent results" }
              ].map((benefit, index) => (
                <div key={index} className="flex items-start gap-4 p-4 bg-[#14273F]/50 rounded-lg border border-[#4CC9F0]/10 hover:border-[#4CC9F0]/30 transition-colors">
                  <div className="mt-1">{benefit.icon}</div>
                  <div>
                    <h4 className="font-semibold text-white mb-1">{benefit.title}</h4>
                    <p className="text-sm text-gray-300">{benefit.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────── Call to Action ──────────────────── */}
      <section className="bg-gradient-to-r from-[#4CC9F0]/10 via-[#3A86FF]/10 to-[#7209B7]/10 py-20">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Ready to Build Smarter Portfolios?
          </h2>
          <p className="text-lg text-gray-300 mb-10 max-w-2xl mx-auto">
            Join the future of portfolio construction with AI-powered strategies, classical algorithms, and comprehensive backtesting tools.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              href="/dashboard"
              className="bg-gradient-to-r from-[#4CC9F0] to-[#3A86FF] hover:from-[#3A86FF] hover:to-[#7209B7] text-[#0D1B2A] font-semibold px-8 py-4 rounded-full transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Try It Now - Free
            </Link>
            {!loading && !user && (
              <Link
                href="/auth/login"
                className="border-2 border-[#4CC9F0] text-white hover:bg-[#4CC9F0] hover:text-[#0D1B2A] px-8 py-4 rounded-full transition-all duration-300 font-semibold"
              >
                Sign In to Save Strategies & History
              </Link>
            )}
          </div>
          <p className="text-sm text-gray-400 mt-6">
            No credit card required • Start building portfolios immediately
          </p>
        </div>
      </section>

      {/* ─────────────────────── Footer ─────────────────────── */}
      <Footer />
    </div>
  );
}