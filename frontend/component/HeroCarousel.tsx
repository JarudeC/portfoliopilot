"use client";

export default function HeroCarousel() {
  return (
    <section className="relative w-full min-h-[90vh] flex items-center justify-center bg-[#0D1B2A] overflow-hidden">
      {/* Curved side masks */}
      <div className="pointer-events-none absolute -left-40 top-0 h-full w-[500px] bg-[#0D1B2A] rounded-r-full" />
      <div className="pointer-events-none absolute -right-40 top-0 h-full w-[500px] bg-[#0D1B2A] rounded-l-full" />

      {/* Hero copy */}
      <div className="relative z-10 max-w-5xl px-6 text-center">
        <h1 className="text-4xl md:text-5xl font-bold mb-6">
          Forecast Smarter. Invest Better.
        </h1>
        <p className="text-lg md:text-xl text-gray-300 mb-8">
          Visualize forecasts, configure algorithms, and build data-driven strategies — all in one dashboard.
        </p>
        <div className="flex justify-center gap-4">
          <a
            href="/dashboard"
            className="bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-6 py-3 rounded-full transition"
          >
            Start Forecasting
          </a>
          <a
            href="/api/auth/signin"
            className="border border-[#4CC9F0] text-white hover:bg-[#14273F] hover:text-[#4CC9F0] px-6 py-3 rounded-full transition"
          >
            Sign In to Save
          </a>
        </div>
      </div>
    </section>
  );
}
