"use client";

import { useState } from "react";
import Navbar from "../../component/Navbar";
import Footer from "../../component/Footer";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

/* Dow 30 – Forecast algos – Back-test algos */
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
  /* `group` lets the chevron rotate on open */
  <details className="group bg-[#14273F] rounded-lg text-white ring-1 ring-[#1B263B]">
    {/* Header row */}
    <summary className="cursor-pointer px-5 py-3 flex items-center justify-between list-none">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>

      {/* Chevron — rotates when open */}
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

    {/* Body */}
    <div className="px-5 py-4 border-t border-[#1B263B]">{children}</div>
  </details>
);

// Type for forecast data per ticker
type ForecastData = {
  historySeries: { date: string; price: number }[];
  forecastSeries: { date: string; price: number }[];
};

export default function Dashboard() {
  /* ─ State ─ */
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
      if (p.includes(t)) return p.filter((x) => x !== t); // uncheck
      if (p.length >= 8) {
        // limit
        window.alert("You can select a maximum of 8 stocks.");
        return p;
      }
      return [...p, t]; // add
    });

  const runBacktest = async () => {
    if (!tickers.length || loading) return; // guard

    setLoading(true);
    setProg(5);
    setNav(null);
    setWeights(null);
    setMetrics(null);

    try {
      /* ① POST /api/train -------------------------------------------------- */
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

      /* ② Poll /api/train/{id} every 2 s ------------------------------- */
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
            pct = Math.min(pct + 6, 95); // ▲ smoother bar
            setProg(pct);
          }
        } catch (e) {
          clearInterval(poll); // ▲ stop on fetch error
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
    console.log(`${ticker} payload:`, payload); // Debug log
    
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

/* merge once at the end */
setForecastDataMap(tempDataMap);
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
      {/* ───── Main grid ───── */}
      <main className="flex-1 pt-[72px] pb-20 px-4 lg:px-16 grid grid-cols-1 lg:grid-cols-[14rem_repeat(2,minmax(0,1fr))] gap-4">
        {/* LEFT ░ Tick list */}
        <aside className="bg-[#14273F] rounded-xl p-6 flex flex-col overflow-y-auto">
          {/* ► Filters container */}
          <h2 className="text-lg font-semibold mb-6">DOW30 Stocks</h2>

          {/* 2. Dow‑30 ticker checklist */}
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

              {/* pop-up */}
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
                    opts={HIST_DAYS}
                  />
                </Filter>
                <Filter label="Forecast Days">
                  <Select
                    value={forecastDays}
                    onChange={(e) => setFcastDays(+e.target.value)}
                    opts={FORECAST_DAYS}
                  />
                </Filter>
              </div>
            </details>
          </div>
          <Filter label="Forecast Model">
            <Select
              value={algo}
              onChange={(e) => setAlgo(e.target.value)}
              opts={FORECAST_ALGOS}
            />
          </Filter>

          {/* ─── Train button ─── */}
          <button
            onClick={runForecast}
            disabled={fLoading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {fLoading ? "Running…" : "Train"}
          </button>

          {/* progress bar */}
          {fLoading && (
            <div className="mt-5 relative h-3 rounded-full bg-[#1B263B]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#3A86FF] to-[#4CC9F0] transition-[width] duration-300"
                style={{ width: `${fProg}%` }}
              />
              <span className="absolute inset-0 flex items-center justify-center text-[11px] font-semibold text-[#E0E8F9]">
                {fProg.toFixed(0)}%
              </span>
            </div>
          )}

          {/* ─── Charts area - multiple small charts ─── */}
          <div className="flex-1 overflow-y-auto mt-6 space-y-4">
            {forecastingTickers.length > 0 &&
            !fLoading &&
            Object.keys(forecastDataMap).length ===
              forecastingTickers.length ? (
              <div className="grid grid-cols-1 gap-4">
                {forecastingTickers.map((ticker) => {
  /* pull THIS ticker’s data */
  const data = forecastDataMap[ticker];

  /* if not yet fetched → placeholder */
  if (!data) {
    return (
      <div key={ticker} className="bg-[#0d1b2a]/50 rounded-lg p-3 h-[172px] flex items-center justify-center">
        <span className="text-xs text-gray-400">Loading…</span>
      </div>
    );
  }

  /* per-ticker series */
  const { historySeries, forecastSeries } = data;
  const allData   = [...historySeries, ...forecastSeries];
  const splitDate = historySeries.at(-1)!.date;
  const tickInt   = Math.max(1, Math.floor(allData.length / 4));

  return (
    <div key={ticker} className="bg-[#0d1b2a]/50 rounded-lg p-3">
      <h4 className="text-sm font-semibold text-cyan-300 mb-2">{ticker}</h4>

      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={allData} margin={{ top: 5, right: 20, bottom: 25, left: 45 }}>
          <XAxis
            dataKey="date"
            interval={tickInt}
            tickFormatter={(d) => {
              const dt = new Date(d);
              return `${(dt.getMonth() + 1).toString().padStart(2, "0")}/${dt
                .getDate()
                .toString()
                .padStart(2, "0")}`;
            }}
            stroke="#7C8BAC"
            fontSize={10}
            tick={{ fill: "#7C8BAC" }}
            axisLine={{ stroke: "#7C8BAC" }}
          />
          <YAxis
            domain={["dataMin - 5", "dataMax + 5"]}
            width={40}
            stroke="#7C8BAC"
            fontSize={10}
            tick={{ fill: "#7C8BAC" }}
            axisLine={{ stroke: "#7C8BAC" }}
            tickFormatter={(v) => v.toFixed(0)}
          />
          <Tooltip
            contentStyle={{
              background: "#1B263B",
              border: "none",
              borderRadius: "4px",
              color: "#E0E8F9",
              fontSize: "12px",
            }}
            formatter={(v: number) => v.toFixed(2)}
            labelFormatter={(l) => l.slice(0, 10)}
          />

          {/* solid blue for full series */}
          <Line
            dataKey="price"
            type="monotone"
            stroke="#4CC9F0"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />

          {/* dotted red separator */}
          <ReferenceLine
            x={splitDate}
            stroke="#FF6B6B"
            strokeDasharray="3 3"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
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
        {/* RIGHT ░ Back-test pane */}
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
                    opts={BTHIST_DAYS}
                  />
                </Filter>
                <Filter label="Look-back Days">
                  <Select
                    value={lookBack}
                    onChange={(e) => setLookBack(+e.target.value)}
                    opts={LOOKBACKS}
                  />
                </Filter>
                <Filter label="Eval Window">
                  <Select
                    value={evalWin}
                    onChange={(e) => setEvalWin(+e.target.value)}
                    opts={EVALWINS}
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
              opts={BACKTEST_ALGOS}
            />
          </Filter>

          <button
            onClick={runBacktest}
            disabled={loading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {loading ? "Running…" : "Train"}
          </button>

          {/* Progress + output placeholders */}
          {loading && (
            <div className="mt-5 relative h-4 rounded-full bg-[#1B263B]">
              {/* filled part */}
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#3A86FF] to-[#4CC9F0] transition-[width] duration-300"
                style={{ width: `${prog}%` }}
              />
              {/* % text */}
              <span className="absolute inset-0 flex items-center justify-center text-[12px] font-semibold text-[#E0E8F9]">
                {prog.toFixed(0)}%
              </span>
            </div>
          )}

          {/* After training finishes you'd conditionally render pie + table */}
          {/* ───── RESULTS ─────────────────────────────────────────── */}
          {!loading && prog === 100 && nav && weights && metrics && (
            <>
              {/* ① Final Portfolio Weights (Pie) */}
              <h3 className="mt-6 mb-2 text-sm font-semibold text-cyan-300">
                Final Portfolio Weights
              </h3>
              <div className="h-52 bg-[#0d1b2a]/50 rounded-xl p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={Object.entries(weights).map(([name, value]) => ({
                        name,
                        value,
                      }))}
                      dataKey="value"
                      nameKey="name"
                      cx="40%" /* leave room on the right for legend */
                      cy="50%"
                      outerRadius={85}
                      innerRadius={38}
                      stroke="#0d1b2a"
                      strokeWidth={2}
                      paddingAngle={2}
                    >
                      {Object.keys(weights).map((_, i) => {
                        /* monochrome-variant palette: cyan → indigo range */
                        const hues = [190, 200, 210, 220, 230, 240];
                        return (
                          <Cell
                            key={i}
                            fill={`hsl(${hues[i % hues.length]} 70% ${
                              55 - i * 3
                            }%)`}
                          />
                        );
                      })}
                    </Pie>
                    <Tooltip
                      formatter={(v: number) => (v * 100).toFixed(1) + "%"}
                      contentStyle={{
                        background: "#1B263B",
                        border: "none",
                        color: "#E0E8F9",
                      }}
                      itemStyle={{ color: "#E0E8F9" }}
                    />
                    <Legend
                      verticalAlign="middle"
                      align="right"
                      layout="vertical"
                      iconType="circle"
                      wrapperStyle={{
                        fontSize: "0.75rem",
                        lineHeight: "1.25rem",
                        color: "#E0E8F9",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* ② NAV / PnL Curve (Line) */}
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
                      tickFormatter={(d) => d.slice(2, 7)} /* YY-MM */
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

              {/* ③ Performance Metrics */}
              <h3 className="mt-8 mb-2 text-sm font-semibold text-cyan-300">
                Performance Metrics
              </h3>
              <div className="bg-[#0d1b2a]/50 rounded-xl p-4 overflow-auto text-sm">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-gray-400">
                      <th className="pb-1">Metric</th>
                      <th className="pb-1 text-right">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Return", metrics.Return],
                      ["Annual Ret.", metrics.AnnualReturn],
                      ["Daily Vol.", metrics.DailyVol],
                      ["Annual Vol.", metrics.AnnualVol],
                      ["Sharpe", metrics.Sharpe],
                      ["Sortino", metrics.Sortino],
                    ].map(([k, v]) => (
                      <tr key={k} className="border-t border-[#1B263B]">
                        <td className="py-1 text-gray-300">{k}</td>
                        <td className="py-1 text-right font-semibold">
                          {typeof v === "number" ? v.toFixed(3) : v ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </aside>
      </main>

      <Footer />
    </div>
  );
}

/* ───── Tiny helpers ───── */
function Filter({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-gray-400">{label}</span>
      {children}
    </div>
  );
}

function Select({
  value,
  onChange,
  opts,
}: {
  value: any;
  onChange: any;
  opts: (string | number)[];
}) {
  return (
    <div className="relative inline-block w-full">
      <select
        value={value}
        onChange={onChange}
        className="select-dark appearance-none pr-8 w-full"
      >
        {opts.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
      <svg
        className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 text-gray-400 transform -translate-y-1/2"
        viewBox="0 0 20 20"
        fill="none"
        stroke="currentColor"
      >
        <path d="M6 8l4 4 4-4" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </div>
  );
}
