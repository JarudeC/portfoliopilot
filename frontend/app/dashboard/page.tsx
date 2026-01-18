/**
 * Main trading dashboard for portfolio forecasting and backtesting.
 * Orchestrates stock selection, algorithm configuration, and results display.
 */
"use client";

import { useState } from "react";
import { Settings } from "lucide-react";
import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";
import { ForecastChart, PortfolioPieChart, MetricsTable } from "../../components/charts";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { ProgressBar, Filter, Select } from "../../components/ui";
import { ErrorBoundary } from "../../components/ui/ErrorBoundary";
import { ClaudePopup } from "../../components/claude";
import { type ForecastData } from "../../lib/utils/forecastMetrics";
import {
  useBacktest,
  useForecast,
  BACKTEST_ALGOS,
  FORECAST_ALGOS,
  LOOKBACKS,
  EVALWINS,
  BTHIST_DAYS,
  HIST_DAYS,
  FORECAST_DAYS,
} from "./hooks";

const DOW30 = [
  "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
  "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
  "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
];

/**
 * Expandable dropdown container for parameter configuration.
 */
const Dropdown = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <details className="group bg-[#14273F] rounded-lg text-white ring-1 ring-[#1B263B]">
    <summary className="cursor-pointer px-5 py-3 flex items-center justify-between list-none">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      <svg
        className="h-4 w-4 shrink-0 transition-transform duration-200 group-open:rotate-180"
        viewBox="0 0 24 24"
        stroke="currentColor"
        fill="none"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </summary>
    <div className="px-5 py-4 border-t border-[#1B263B]">{children}</div>
  </details>
);

/**
 * Individual forecast stock item with mini chart.
 */
const ForecastStockItem = ({ ticker, data }: { ticker: string; data: ForecastData }) => (
  <div>
    <h4 className="text-sm font-semibold text-cyan-300 mb-2">{ticker}</h4>
    <ForecastChart
      historySeries={data.historySeries}
      forecastSeries={data.forecastSeries}
      height={140}
      showTitle={false}
    />
  </div>
);

export default function Dashboard() {
  const [tickers, setTickers] = useState<string[]>([]);

  const backtest = useBacktest();
  const forecast = useForecast();

  const toggle = (t: string) =>
    setTickers((p) => {
      if (p.includes(t)) return p.filter((x) => x !== t);
      if (p.length >= 8) {
        window.alert("You can select a maximum of 8 stocks.");
        return p;
      }
      return [...p, t];
    });

  /**
   * Convert selected tickers to StockData format for Claude strategies.
   */
  const getStockDataFromTickers = () =>
    tickers.map(ticker => ({
      symbol: ticker,
      price: 100 + Math.random() * 200,
      marketCap: 1000000000 + Math.random() * 2000000000000,
      volume: 1000000 + Math.random() * 50000000
    }));

  return (
    <div className="min-h-screen bg-[#0D1B2A] text-white flex flex-col">
      <Navbar />
      <main id="hero"></main>

      {/* Three-column layout: stocks, forecasting, backtesting */}
      <main className="flex-1 pt-[72px] pb-20 px-4 lg:px-16 grid grid-cols-1 lg:grid-cols-[14rem_repeat(2,minmax(0,1fr))] gap-4">

        {/* Stock Selection Sidebar */}
        <aside className="bg-[#14273F] rounded-xl p-6 flex flex-col overflow-y-auto">
          <h2 className="text-lg font-semibold mb-6">DOW30 Stocks</h2>
          <p className="text-xs text-gray-400 mb-2">
            Select up to <span className="text-[#4CC9F0] font-semibold">8</span> stocks
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
              <label htmlFor="unselect" className="text-sm text-white cursor-pointer font-semibold">
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
                <label htmlFor={t} className="cursor-pointer">{t}</label>
              </li>
            ))}
          </ul>
        </aside>

        {/* Forecasting Panel */}
        <section className="flex flex-col bg-[#14273F] rounded-xl p-6 h-full">
          <div className="flex items-start justify-between mb-6">
            <h2 className="text-lg font-semibold">Forecasting</h2>
            <details className="relative group">
              <summary className="cursor-pointer text-sm flex items-center gap-1 select-none">
                Parameters
                <svg className="h-4 w-4 transition-transform group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                  <path d="M6 8l4 4 4-4" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </summary>
              <div className="param-pop absolute right-0 mt-2 space-y-4 z-10">
                <button
                  onClick={forecast.resetParams}
                  className="text-xs text-[#4CC9F0] hover:text-[#3A86FF] transition"
                >
                  Reset Defaults
                </button>
                <Filter label="History Days (Trading)">
                  <Select
                    value={forecast.params.histDays}
                    onChange={(e) => forecast.setParams(p => ({ ...p, histDays: +e.target.value }))}
                    options={HIST_DAYS}
                  />
                </Filter>
                <Filter label="Forecast Days">
                  <Select
                    value={forecast.params.forecastDays}
                    onChange={(e) => forecast.setParams(p => ({ ...p, forecastDays: +e.target.value }))}
                    options={FORECAST_DAYS}
                  />
                </Filter>
              </div>
            </details>
          </div>

          <Filter label="Forecast Model">
            <div className="flex items-center gap-2">
              <Select
                value={forecast.algo}
                onChange={forecast.handleAlgoChange}
                options={FORECAST_ALGOS}
              />
              {forecast.algo === "Custom AI Strategy" && (
                <button
                  onClick={forecast.openPopup}
                  className="p-2 text-[#4CC9F0] hover:bg-[#4CC9F0]/10 rounded-lg transition-colors"
                  title="Configure AI Strategy"
                >
                  <Settings className="w-4 h-4" />
                </button>
              )}
            </div>
          </Filter>

          <button
            onClick={() => forecast.runForecast(tickers)}
            disabled={forecast.loading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {forecast.loading ? "Running…" : "Train"}
          </button>

          {forecast.loading && <ProgressBar progress={forecast.progress} className="mt-5 h-3" />}

          {/* Forecast Results */}
          <div className="flex-1 overflow-y-auto mt-6 space-y-4">
            {forecast.forecastingTickers.length > 0 &&
              !forecast.loading &&
              Object.keys(forecast.forecastDataMap).length === forecast.forecastingTickers.length ? (
              <div className="grid grid-cols-1 gap-4">
                {forecast.forecastingTickers.map((ticker) => {
                  const data = forecast.forecastDataMap[ticker];
                  if (!data) {
                    return (
                      <div key={ticker} className="bg-[#0d1b2a]/50 rounded-lg p-3 h-[172px] flex items-center justify-center">
                        <span className="text-xs text-gray-400">Loading…</span>
                      </div>
                    );
                  }
                  return <ForecastStockItem key={ticker} ticker={ticker} data={data} />;
                })}

                {/* Overall Metrics */}
                {forecast.overallMetricsLoading ? (
                  <div className="mt-4">
                    <h4 className="text-sm font-semibold text-cyan-300 mb-3">Performance Metrics</h4>
                    <div className="text-center py-4">
                      <span className="text-xs text-gray-400">Calculating overall metrics...</span>
                    </div>
                  </div>
                ) : forecast.overallMetrics ? (
                  <div className="mt-4">
                    <h4 className="text-sm font-semibold text-cyan-300 mb-3">Performance Metrics</h4>
                    <MetricsTable
                      metrics={{ mse: forecast.overallMetrics.mse, mae: forecast.overallMetrics.mae }}
                      showTitle={false}
                    />
                    <div className="text-xs text-gray-500 mt-2">
                      Total predictions: {forecast.overallMetrics.totalPredictions} | Stocks analyzed: {forecast.overallMetrics.stockCount}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <span className="text-gray-500">
                  {forecast.loading
                    ? `Fetching forecasts for ${forecast.forecastingTickers.length} stocks...`
                    : "Select stocks and run forecast"}
                </span>
              </div>
            )}
          </div>
        </section>

        {/* Backtesting Panel */}
        <aside className="bg-[#14273F] rounded-xl p-6 flex flex-col">
          <div className="flex items-start justify-between mb-6">
            <h2 className="text-lg font-semibold">Run Back-test</h2>
            <details className="relative group">
              <summary className="cursor-pointer text-sm flex items-center gap-1 select-none">
                Parameters
                <svg className="h-4 w-4 transition-transform group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                  <path d="M6 8l4 4 4-4" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </summary>
              <div className="absolute right-0 param-pop space-y-4 z-10">
                <button
                  onClick={backtest.resetParams}
                  className="text-xs text-[#4CC9F0] hover:text-[#3A86FF] transition"
                >
                  Reset Defaults
                </button>
                <Filter label="Back-test Days (Trading)">
                  <Select
                    value={backtest.params.btHistDays}
                    onChange={(e) => backtest.setParams(p => ({ ...p, btHistDays: +e.target.value }))}
                    options={BTHIST_DAYS}
                  />
                </Filter>
                <Filter label="Look-back Days">
                  <Select
                    value={backtest.params.lookBack}
                    onChange={(e) => backtest.setParams(p => ({ ...p, lookBack: +e.target.value }))}
                    options={LOOKBACKS}
                  />
                </Filter>
                <Filter label="Eval Window">
                  <Select
                    value={backtest.params.evalWin}
                    onChange={(e) => backtest.setParams(p => ({ ...p, evalWin: +e.target.value }))}
                    options={EVALWINS}
                  />
                </Filter>
                <Filter label="Transaction Cost">
                  <input
                    type="number"
                    step="0.0001"
                    value={backtest.params.tc}
                    onChange={(e) => backtest.setParams(p => ({ ...p, tc: +e.target.value }))}
                    className="select-dark w-24"
                  />
                </Filter>
              </div>
            </details>
          </div>

          <Filter label="Back-test Model">
            <div className="flex items-center gap-2">
              <Select
                value={backtest.btAlgo}
                onChange={backtest.handleAlgoChange}
                options={BACKTEST_ALGOS}
              />
              {backtest.btAlgo === "Custom AI Strategy" && (
                <button
                  onClick={backtest.openPopup}
                  className="p-2 text-[#4CC9F0] hover:bg-[#4CC9F0]/10 rounded-lg transition-colors"
                  title="Configure AI Strategy"
                >
                  <Settings className="w-4 h-4" />
                </button>
              )}
            </div>
          </Filter>

          <button
            onClick={() => backtest.runBacktest(tickers)}
            disabled={backtest.loading || !tickers.length}
            className="mt-6 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold rounded-full py-2 transition disabled:opacity-40"
          >
            {backtest.loading ? "Running…" : "Train"}
          </button>

          {backtest.loading && <ProgressBar progress={backtest.progress} className="mt-5" />}

          {/* Backtest Results */}
          {!backtest.loading && backtest.progress === 100 && backtest.results.nav && backtest.results.weights && backtest.results.metrics && (
            <div className="space-y-6 mt-6">
              <div>
                <h3 className="mb-2 text-sm font-semibold text-cyan-300">Final Portfolio Weights</h3>
                <PortfolioPieChart weights={backtest.results.weights} height={200} showTitle={false} />
              </div>

              <h3 className="mt-8 mb-2 text-sm font-semibold text-cyan-300">Equity Curve (PnL)</h3>
              <div className="h-56 bg-[#0d1b2a]/50 rounded-xl p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={Object.entries(backtest.results.nav).map(([date, val]) => ({ date, val }))}
                    margin={{ top: 5, right: 18, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="date"
                      tickFormatter={(d) => d.slice(2, 7)}
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
                      contentStyle={{
                        background: "#1B263B",
                        border: "none",
                        borderRadius: "4px",
                        color: "#E0E8F9",
                        fontSize: "12px",
                      }}
                      labelFormatter={(d) => d.slice(0, 10)}
                      formatter={(v: number) => v.toFixed(4)}
                    />
                    <Line type="monotone" dataKey="val" stroke="#4CC9F0" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h3 className="mb-2 text-sm font-semibold text-cyan-300">Performance Metrics</h3>
                <MetricsTable metrics={backtest.results.metrics} showTitle={false} />
              </div>
            </div>
          )}
        </aside>
      </main>

      <Footer />

      {/* Claude Strategy Popups */}
      <ErrorBoundary
        fallback={
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300">
            <h3 className="font-medium mb-2">Strategy Generation Error</h3>
            <p className="text-sm">The strategy generation popup encountered an error. Please refresh the page.</p>
          </div>
        }
      >
        <ClaudePopup
          isOpen={forecast.showPopup}
          onClose={forecast.handlePopupClose}
          mode="forecast"
          stockData={getStockDataFromTickers()}
          onStrategyGenerated={forecast.handleClaudeGenerated}
          onError={forecast.handleClaudeError}
          dashboardParams={{
            historyDays: forecast.params.histDays,
            forecastDays: forecast.params.forecastDays,
            algorithm: forecast.algo
          }}
        />
      </ErrorBoundary>

      <ErrorBoundary
        fallback={
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300">
            <h3 className="font-medium mb-2">Strategy Generation Error</h3>
            <p className="text-sm">The strategy generation popup encountered an error. Please refresh the page.</p>
          </div>
        }
      >
        <ClaudePopup
          isOpen={backtest.showPopup}
          onClose={backtest.handlePopupClose}
          mode="backtest"
          stockData={getStockDataFromTickers()}
          onStrategyGenerated={backtest.handleClaudeGenerated}
          onError={backtest.handleClaudeError}
          dashboardParams={{
            backtestDays: backtest.params.btHistDays,
            lookbackDays: backtest.params.lookBack,
            evaluationWindow: backtest.params.evalWin,
            transactionCost: backtest.params.tc,
            algorithm: backtest.btAlgo
          }}
        />
      </ErrorBoundary>
    </div>
  );
}
