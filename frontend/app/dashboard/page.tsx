// Main trading dashboard for portfolio forecasting and backtesting
"use client";

import { useState } from "react";
import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";
import { ForecastChart, EquityChart, PortfolioPieChart, MetricsTable } from "../../components/charts";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { ProgressBar, Filter, Select } from "../../components/ui";

// Available stock tickers and algorithm options
const DOW30 = [
  "AAPL",
  "AMGN",
  "AXP",
  "BA",
  "CAT",
  "CRM",
  "CSCO",
  "CVX",
  "DIS",
  "DOW",
  "GS",
  "HD",
  "HON",
  "IBM",
  "INTC",
  "JNJ",
  "JPM",
  "KO",
  "MCD",
  "MMM",
  "MRK",
  "MSFT",
  "NKE",
  "PG",
  "TRV",
  "UNH",
  "V",
  "VZ",
  "WBA",
  "WMT",
];
const FORECAST_ALGOS = ["ARIMA", "LSTM", "Autoformer"];
const BACKTEST_ALGOS = ["Naive Markowitz", "GVMP", "PPN", "Margin Trader"];
const LOOKBACKS = [30, 60, 90];
const EVALWINS = [5, 10, 15];
const HIST_DAYS = [60, 90, 180, 365];
const FORECAST_DAYS = [5, 7, 14, 30];
const BTHIST_DAYS = [365, 1095, 1825];

const Dropdown = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  // Expandable dropdown with rotating chevron
  <details className="group bg-[#14273F] rounded-lg text-white ring-1 ring-[#1B263B]">
    {/* Dropdown header with title and chevron */}
    <summary className="cursor-pointer px-5 py-3 flex items-center justify-between list-none">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>

      {/* Chevron rotates 180deg when dropdown opens */}
      <svg
        className="h-4 w-4 shrink-0 transition-transform duration-200 group-open:rotate-180"
        viewBox="0 0 24 24"
        stroke="currentColor"
        fill="none"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </summary>

    {/* Dropdown content area */}
    <div className="px-5 py-4 border-t border-[#1B263B]">{children}</div>
  </details>
);

// Type definition for individual ticker forecast data
type ForecastData = {
  historySeries: { date: string; price: number }[];
  forecastSeries: { date: string; price: number }[];
};

export default function Dashboard() {
  // Component state management
  const [tickers, setTickers] = useState<string[]>([]);
  const [algo, setAlgo] = useState(FORECAST_ALGOS[0]);
  const [btAlgo, setBtAlgo] = useState(BACKTEST_ALGOS[0]);
  const [lookBack, setLookBack] = useState(30);
  const [evalWin, setEvalWin] = useState(5);
  const [tc, setTc] = useState(0.002);
  const [loading, setLoading] = useState(false);
  const [prog, setProg] = useState(0);
  const [histDays, setHistDays] = useState(180);
  const [forecastDays, setFcastDays] = useState(14);
  const [nav, setNav] = useState<Record<string, number> | null>(null);
  const [weights, setWeights] = useState<Record<string, number> | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [btHistDays, setBtHistDays] = useState(365);
  const [fLoading, setFLoading] = useState(false);
  const [fProg, setFProg] = useState(0);
  // Changed to store forecast data for multiple tickers
  const [forecastDataMap, setForecastDataMap] = useState<
    Record<string, ForecastData>
  >({});
  const [forecastingTickers, setForecastingTickers] = useState<string[]>([]);

  const toggle = (t: string) =>
    setTickers((p) => {
      if (p.includes(t)) return p.filter((x) => x !== t); // Remove if already selected
      if (p.length >= 8) {
        // Enforce max selection limit
        window.alert("You can select a maximum of 8 stocks.");
        return p;
      }
      return [...p, t]; // Add to selection
    });

  const runBacktest = async () => {
    if (!tickers.length || loading) return; // Prevent multiple simultaneous runs

    setLoading(true);
    setProg(5);
    setNav(null);
    setWeights(null);
    setMetrics(null);

    try {
      // Start training job on backend
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algo: btAlgo,
          tickers,
          hist_days: btHistDays,
          lookback: lookBack,
          eval_win: evalWin,
          eta: 0.02,
          tc,
        }),
      });
      if (!res.ok) throw new Error(`Backend ${res.status}`);
      const { job_id } = await res.json();
      if (!job_id) throw new Error("No job_id returned");

      // Poll training progress every 2 seconds
      let pct = 8;
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`/api/train/${job_id}`);
          const data = await r.json();

          if (data.status === "done") {
            clearInterval(poll);
            setNav(data.nav);
            setWeights(data.weights);
            setMetrics(data.metrics);
            setProg(100);
            setLoading(false);
          } else if (data.status === "error") {
            clearInterval(poll);
            alert(data.detail || "Training failed");
            setLoading(false);
          } else {
            pct = Math.min(pct + 6, 95); // Increment progress smoothly
            setProg(pct);
          }
        } catch (e) {
          clearInterval(poll); // Stop polling on connection error
          console.error(e);
          alert("Lost connection to backend");
          setLoading(false);
        }
      }, 2000);
    } catch (e) {
      console.error(e);
      alert((e as Error).message);
      setLoading(false);
    }
  };

  const runForecast = async () => {
    if (!tickers.length || fLoading) return;

    setFLoading(true);
    setFProg(0);
    setForecastDataMap({});
    setForecastingTickers([...tickers]);

    const today = new Date();
    const maxEndDate = new Date("2024-12-31");
    const endDate = today > maxEndDate ? maxEndDate : today;
    const end = endDate.toISOString().slice(0, 10);
    const start = new Date(endDate.getTime() - histDays * 86_400_000)
      .toISOString()
      .slice(0, 10);

    const totalTickers = tickers.length;
    let completedTickers = 0;
    const tempDataMap: Record<string, ForecastData> = {};

    try {
      for (const ticker of tickers) {
  try {
    const res = await fetch(`/api/forecast/${algo.toLowerCase()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, start, end, horizon: forecastDays }),
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const payload = await res.json();
    
    // Check if we got the expected data structure
    if (!payload.history_dates || !payload.history_values || 
        !payload.forecast_dates || !payload.forecast_values) {
      console.error(`Invalid payload structure for ${ticker}:`, payload);
      throw new Error(`Invalid response structure: missing required fields`);
    }

    const toSeries = (d: string[], v: number[]) =>
      d.map((x, i) => ({ date: x, price: v[i] }));

    tempDataMap[ticker] = {
      historySeries: toSeries(payload.history_dates, payload.history_values),
      forecastSeries: toSeries(
        payload.forecast_dates,
        payload.forecast_values,
      ),
    };

    completedTickers++;
    setFProg((completedTickers / tickers.length) * 100);
  } catch (err) {
    console.error(`forecast ${ticker}:`, err);
    // Add empty data for failed ticker to prevent UI crashes
    tempDataMap[ticker] = {
      historySeries: [],
      forecastSeries: [],
    };
    // Continue processing other tickers even if this one fails
    completedTickers++;
    setFProg((completedTickers / tickers.length) * 100);
  }
}

// Update state with all forecast results at once
setForecastDataMap(tempDataMap);

// Save forecast session to database for history tracking
try {
  const forecasts = Object.entries(tempDataMap).filter(([_, data]) => 
    data.historySeries.length > 0 && data.forecastSeries.length > 0
  );
  
  if (forecasts.length > 0) {
    const logPayload = {
      type: 'forecast',
      stocks: forecasts.map(([ticker, _]) => ticker),
      model: algo.toUpperCase(),
      parameters: {
        algorithm: algo,
        start_date: start,
        end_date: end,
        history_days: histDays,
        forecast_days: forecastDays,
        tickers: forecasts.map(([ticker, _]) => ticker)
      },
      results: {
        predictions: forecasts.flatMap(([ticker, data]) => 
          data.forecastSeries.map(point => ({
            ticker,
            date: point.date,
            price: point.price
          }))
        )
      },
      charts: Object.fromEntries(forecasts.map(([ticker, data]) => [
        ticker, {
          history: data.historySeries,
          forecast: data.forecastSeries
        }
      ]))
    };
    
    // Send forecast data to logging endpoint
    await fetch('/api/forecast/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(logPayload)
    });
    
  }
} catch (logError) {
  console.error('Failed to log combined forecast:', logError);
  // Continue even if logging fails
}

setFLoading(false);
    } catch (e) {
      console.error(e);
      alert((e as Error).message);
      setFLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0D1B2A] text-white flex flex-col">
      <Navbar />
      <main id="hero"></main>
      {/* Three-column layout: stocks, forecasting, backtesting */}
      <main className="flex-1 pt-[72px] pb-20 px-4 lg:px-16 grid grid-cols-1 lg:grid-cols-[14rem_repeat(2,minmax(0,1fr))] gap-4">
        {/* Left sidebar: stock selection */}
        <aside className="bg-[#14273F] rounded-xl p-6 flex flex-col overflow-y-auto">
          {/* Stock selection filters */}
          <h2 className="text-lg font-semibold mb-6">DOW30 Stocks</h2>

          {/* DOW 30 stock checkboxes */}
          <p className="text-xs text-gray-400 mb-2">
            Select up to <span className="text-[#4CC9F0] font-semibold">8</span>{" "}
            stocks
          </p>
          <ul className="space-y-2 text-sm">
            <div className="flex items-center gap-2 mb-2">
              <input
                id="unselect"
                type="checkbox"
                className="accent-[#4CC9F0]"
                checked={false}
                onChange={() => setTickers([])}
              />
              <label
                htmlFor="unselect"
                className="text-sm text-white cursor-pointer font-semibold"
              >
                (Unselect All)
              </label>
            </div>

            {DOW30.map((t) => (
              <li key={t} className="flex items-center gap-2">
                <input
                  id={t}
                  type="checkbox"
                  className="accent-[#4CC9F0]"
                  checked={tickers.includes(t)}
                  onChange={() => toggle(t)}
                />
                <label htmlFor={t} className="cursor-pointer">
                  {t}
                </label>
              </li>
            ))}
          </ul>
        </aside>
        <section className="flex flex-col bg-[#14273F] rounded-xl p-6 h-full">
          <div className="flex items-start justify-between mb-6">
            <h2 className="text-lg font-semibold">Forecasting</h2>
            <details className="relative group">
              <summary className="cursor-pointer text-sm flex items-center gap-1 select-none">
                Parameters
                <svg
                  className="h-4 w-4 transition-transform group-open:rotate-180"
                  viewBox="0 0 20 20"
                  fill="none"
                  stroke="currentColor"
                >
                  <path
                    d="M6 8l4 4 4-4"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </summary>

              {/* Parameter configuration popup */}
              <div className="param-pop absolute right-0 mt-2 space-y-4 z-10">
                <button
                  onClick={() => {
                    setHistDays(180);
                    setFcastDays(14);
                  }}
                  className="text-xs text-[#4CC9F0] hover:text-[#3A86FF] transition"
                >
                  Reset Defaults
                </button>
                <Filter label="History Days">
                  <Select
                    value={histDays}
                    onChange={(e) => setHistDays(+e.target.value)}
                    options={HIST_DAYS}
                  />
                </Filter>
                <Filter label="Forecast Days">
                  <Select
                    value={forecastDays}
                    onChange={(e) => setFcastDays(+e.target.value)}
                    options={FORECAST_DAYS}
                  />
                </Filter>
              </div>
            </details>
          </div>
          <Filter label="Forecast Model">
            <Select
              value={algo}
              onChange={(e) => setAlgo(e.target.value)}
              options={FORECAST_ALGOS}
            />
          </Filter>

          {/* Start forecasting button */}
          <button
            onClick={runForecast}
            disabled={fLoading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {fLoading ? "Running…" : "Train"}
          </button>

          {/* Progress indicator during training */}
          {fLoading && (
            <ProgressBar progress={fProg} className="mt-5 h-3" />
          )}

          {/* Forecast results visualization area */}
          <div className="flex-1 overflow-y-auto mt-6 space-y-4">
            {forecastingTickers.length > 0 &&
            !fLoading &&
            Object.keys(forecastDataMap).length ===
              forecastingTickers.length ? (
              <div className="grid grid-cols-1 gap-4">
                {forecastingTickers.map((ticker) => {
                  const data = forecastDataMap[ticker];

                  if (!data) {
                    return (
                      <div key={ticker} className="bg-[#0d1b2a]/50 rounded-lg p-3 h-[172px] flex items-center justify-center">
                        <span className="text-xs text-gray-400">Loading…</span>
                      </div>
                    );
                  }

                  const { historySeries, forecastSeries } = data;

                  return (
                    <div key={ticker}>
                      <h4 className="text-sm font-semibold text-cyan-300 mb-2">{ticker}</h4>
                      <ForecastChart
                        historySeries={historySeries}
                        forecastSeries={forecastSeries}
                        height={140}
                        showTitle={false}
                      />
                    </div>
                  );
                })}

              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <span className="text-gray-500">
                  {fLoading
                    ? `Fetching forecasts for ${forecastingTickers.length} stocks...`
                    : "Select stocks and run forecast"}
                </span>
              </div>
            )}
          </div>
        </section>
        {/* Right sidebar: backtesting controls and results */}
        <aside className="bg-[#14273F] rounded-xl p-6 flex flex-col">
          <div className="flex items-start justify-between mb-6">
            <h2 className="text-lg font-semibold">Run Back-test</h2>

            <details className="relative group">
              <summary className="cursor-pointer text-sm flex items-center gap-1 select-none">
                Parameters
                <svg
                  className="h-4 w-4 transition-transform group-open:rotate-180"
                  viewBox="0 0 20 20"
                  fill="none"
                  stroke="currentColor"
                >
                  <path
                    d="M6 8l4 4 4-4"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </summary>

              <div className="absolute right-0 param-pop space-y-4 z-10">
                <button
                  onClick={() => {
                    setBtHistDays(365);
                    setLookBack(30);
                    setEvalWin(5);
                    setTc(0.002);
                  }}
                  className="text-xs text-[#4CC9F0] hover:text-[#3A86FF] transition"
                >
                  Reset Defaults
                </button>
                <Filter label="Back-test Days">
                  <Select
                    value={btHistDays}
                    onChange={(e) => setBtHistDays(+e.target.value)}
                    options={BTHIST_DAYS}
                  />
                </Filter>
                <Filter label="Look-back Days">
                  <Select
                    value={lookBack}
                    onChange={(e) => setLookBack(+e.target.value)}
                    options={LOOKBACKS}
                  />
                </Filter>
                <Filter label="Eval Window">
                  <Select
                    value={evalWin}
                    onChange={(e) => setEvalWin(+e.target.value)}
                    options={EVALWINS}
                  />
                </Filter>
                <Filter label="Transaction Cost">
                  <input
                    type="number"
                    step="0.0001"
                    value={tc}
                    onChange={(e) => setTc(+e.target.value)}
                    className="select-dark w-24"
                  />
                </Filter>
              </div>
            </details>
          </div>
          <Filter label="Back-test Model">
            <Select
              value={btAlgo}
              onChange={(e) => setBtAlgo(e.target.value)}
              options={BACKTEST_ALGOS}
            />
          </Filter>

          <button
            onClick={runBacktest}
            disabled={loading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {loading ? "Running…" : "Train"}
          </button>

          {/* Training progress and results */}
          {loading && (
            <ProgressBar progress={prog} className="mt-5" />
          )}

          {/* Backtest results visualization */}
          {!loading && prog === 100 && nav && weights && metrics && (
            <div className="space-y-6 mt-6">
              <div>
                <h3 className="mb-2 text-sm font-semibold text-cyan-300">
                  Final Portfolio Weights
                </h3>
                <PortfolioPieChart weights={weights} height={200} showTitle={false} />
              </div>

              {/* Portfolio equity curve over time */}
              <h3 className="mt-8 mb-2 text-sm font-semibold text-cyan-300">
                Equity Curve (PnL)
              </h3>
              <div className="h-56 bg-[#0d1b2a]/50 rounded-xl p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={Object.entries(nav).map(([date, val]) => ({
                      date,
                      val,
                    }))}
                    margin={{ top: 5, right: 18, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="date"
                      tickFormatter={(d) => d.slice(2, 7)} /* Format dates as YY-MM */
                      minTickGap={40}
                      stroke="#7C8BAC"
                      fontSize={12}
                    />
                    <YAxis
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(v) => v.toFixed(2)}
                      stroke="#7C8BAC"
                      fontSize={12}
                    />
                    <Tooltip
                      labelFormatter={(d) => d}
                      formatter={(v: number) => v.toFixed(4)}
                    />
                    <Line
                      type="monotone"
                      dataKey="val"
                      stroke="#4CC9F0"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h3 className="mb-2 text-sm font-semibold text-cyan-300">
                  Performance Metrics
                </h3>
                <MetricsTable metrics={metrics} showTitle={false} />
              </div>
            </div>
          )}
        </aside>
      </main>

      <Footer />
    </div>
  );
}
