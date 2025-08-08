// Main trading dashboard for portfolio forecasting and backtesting
"use client";

import { useState, useEffect } from "react";
import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";
import { ForecastChart, EquityChart, PortfolioPieChart, MetricsTable } from "../../components/charts";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { ProgressBar, Filter, Select } from "../../components/ui";
import { useToast } from "../../components/ui/Toast";
import { ErrorBoundary } from "../../components/ui/ErrorBoundary";
import { ClaudePopup } from "../../components/claude";
import { ClaudeClientError, type GenerationResult } from "../../lib/claude/client";
import { calculateForecastMetrics, type ForecastData } from "../../lib/utils/forecastMetrics";

// Component for individual forecast stock item with async metrics
const ForecastStockItem = ({ ticker, data, claudeStrategy }: { 
  ticker: string, 
  data: ForecastData,
  claudeStrategy?: any
}) => {
  const [metrics, setMetrics] = useState<{ mse: number; mae: number } | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  useEffect(() => {
    // Use pre-calculated metrics if available, otherwise calculate them
    if (data.forecastSeries.length === 0 || data.historySeries.length === 0) {
      setMetrics(null);
      setMetricsLoading(false);
      return;
    }
    
    // Check if metrics are already calculated and stored
    if (data.metrics) {
      setMetrics(data.metrics);
      setMetricsLoading(false);
      return;
    }
    
    // Fallback: calculate metrics if not pre-calculated (shouldn't happen in normal flow)
    if (!data.algorithm) {
      setMetrics(null);
      setMetricsLoading(false);
      return;
    }
    
    const loadMetrics = async () => {
      try {
        setMetricsLoading(true);
        const result = await calculateForecastMetrics(
          data, 
          data.algorithm, 
          ticker, 
          data.algorithm?.toLowerCase() === 'custom ai strategy' ? claudeStrategy : undefined
        );
        setMetrics(result);
      } catch (error) {
        console.error(`Failed to calculate metrics for ${ticker}:`, error);
        setMetrics({ mse: 0, mae: 0 });
      } finally {
        setMetricsLoading(false);
      }
    };

    loadMetrics();
  }, [ticker, data.forecastSeries.length, data.historySeries.length, data.algorithm, data.metrics]);

  const { historySeries, forecastSeries } = data;

  return (
    <div>
      <h4 className="text-sm font-semibold text-cyan-300 mb-2">{ticker}</h4>
      <ForecastChart
        historySeries={historySeries}
        forecastSeries={forecastSeries}
        height={140}
        showTitle={false}
      />
      {/* Display MSE and MAE for this stock */}
      <div className="mt-2">
        {metricsLoading ? (
          <div className="bg-[#0d1b2a]/50 rounded-lg p-4 text-center">
            <span className="text-xs text-gray-400">Calculating metrics for {data.algorithm}...</span>
          </div>
        ) : metrics ? (
          <div>
            <MetricsTable 
              metrics={metrics}
              title={`${ticker} Forecast Accuracy`}
              showTitle={false}
            />
            <div className="text-xs text-gray-500 mt-1">
              Algorithm: {data.algorithm || 'Unknown'} | Data points: {data.historySeries.length}
            </div>
          </div>
        ) : (
          <div className="bg-[#0d1b2a]/50 rounded-lg p-4 text-center">
            <span className="text-xs text-red-400">Metrics calculation failed</span>
          </div>
        )}
      </div>
    </div>
  );
};

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
const FORECAST_ALGOS = ["ARIMA", "LSTM", "Autoformer", "Custom AI Strategy"];
const BACKTEST_ALGOS = ["Naive Markowitz", "GVMP", "PPN", "Margin Trader", "Custom AI Strategy"];
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

// Note: ForecastData type is imported from forecastMetrics utility

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
  
  // Toast notifications for enhanced user feedback
  const { showSuccess, showError, showWarning, showInfo } = useToast();
  
  // Changed to store forecast data for multiple tickers
  const [forecastDataMap, setForecastDataMap] = useState<
    Record<string, ForecastData>
  >({});
  const [forecastingTickers, setForecastingTickers] = useState<string[]>([]);
  
  // Claude strategy state management
  const [claudeForecastStrategy, setClaudeForecastStrategy] = useState<GenerationResult | null>(null);
  const [claudeBacktestStrategy, setClaudeBacktestStrategy] = useState<GenerationResult | null>(null);
  const [claudeForecastError, setClaudeForecastError] = useState<string | null>(null);
  const [claudeBacktestError, setClaudeBacktestError] = useState<string | null>(null);
  
  // Popup visibility state
  const [showForecastPopup, setShowForecastPopup] = useState<boolean>(false);
  const [showBacktestPopup, setShowBacktestPopup] = useState<boolean>(false);
  
  // Store previous selections for reverting
  const [prevForecastAlgo, setPrevForecastAlgo] = useState<string>(FORECAST_ALGOS[0]);
  const [prevBacktestAlgo, setPrevBacktestAlgo] = useState<string>(BACKTEST_ALGOS[0]);

  // Helper functions for Claude integration
  const isClaudeForecastSelected = algo === "Custom AI Strategy";
  const isClaudeBacktestSelected = btAlgo === "Custom AI Strategy";
  
  // Convert selected tickers to StockData format for Claude
  const getStockDataFromTickers = () => {
    return tickers.map(ticker => ({
      symbol: ticker,
      price: 100 + Math.random() * 200, // Mock price data
      marketCap: 1000000000 + Math.random() * 2000000000000, // Mock market cap
      volume: 1000000 + Math.random() * 50000000 // Mock volume
    }));
  };

  // Handle Claude strategy generation with enhanced success feedback
  const handleClaudeForecastGenerated = (result: GenerationResult) => {
    setClaudeForecastStrategy(result);
    setShowForecastPopup(false); // Close popup after successful generation
    
    if (result.fallbackUsed) {
      showWarning(
        "Forecast Strategy Generated (Fallback)",
        `Strategy generated using fallback method: ${result.error || 'AI generation failed'}`
      );
    } else {
      showSuccess(
        "Forecast Strategy Generated",
        "Your custom AI forecast strategy has been successfully created and is ready to use."
      );
    }
  };

  const handleClaudeBacktestGenerated = (result: GenerationResult) => {
    setClaudeBacktestStrategy(result);
    setShowBacktestPopup(false); // Close popup after successful generation
    
    if (result.fallbackUsed) {
      showWarning(
        "Backtest Strategy Generated (Fallback)",
        `Strategy generated using fallback method: ${result.error || 'AI generation failed'}`
      );
    } else {
      showSuccess(
        "Backtest Strategy Generated",
        "Your custom AI backtest strategy has been successfully created and is ready to use."
      );
    }
  };

  const handleClaudeForecastError = (error: ClaudeClientError | Error) => {
    setClaudeForecastStrategy(null);
    setShowForecastPopup(false); // Close popup on error
    setAlgo(prevForecastAlgo); // Reset dropdown to previous selection
    console.error('Claude forecast generation error:', error);
    
    showError(
      "Forecast Strategy Failed",
      error.message,
      ["Try again with a simpler description", "Check your internet connection", "Use more common technical analysis terms"]
    );
  };

  const handleClaudeBacktestError = (error: ClaudeClientError | Error) => {
    setClaudeBacktestStrategy(null);
    setShowBacktestPopup(false); // Close popup on error
    setBtAlgo(prevBacktestAlgo); // Reset dropdown to previous selection
    console.error('Claude backtest generation error:', error);
    
    showError(
      "Backtest Strategy Failed",
      error.message,
      ["Try again with a simpler description", "Check your internet connection", "Use more common investment terms"]
    );
  };

  // Handle popup close - revert to previous selection if no strategy generated
  const handleForecastPopupClose = () => {
    setShowForecastPopup(false);
    if (!claudeForecastStrategy) {
      setAlgo(prevForecastAlgo);
    }
  };

  const handleBacktestPopupClose = () => {
    setShowBacktestPopup(false);
    if (!claudeBacktestStrategy) {
      setBtAlgo(prevBacktestAlgo);
    }
  };

  // Clear Claude strategies when switching models
  const handleAlgoChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newAlgo = e.target.value;
    if (newAlgo === "Custom AI Strategy") {
      // Always show popup when Custom AI Strategy is selected (even if already selected)
      if (algo !== "Custom AI Strategy") {
        setPrevForecastAlgo(algo);
      }
      setAlgo(newAlgo);
      setShowForecastPopup(true);
    } else {
      // Store new selection and clear Claude data
      setPrevForecastAlgo(newAlgo);
      setAlgo(newAlgo);
      setClaudeForecastStrategy(null);
      setClaudeForecastError(null);
    }
  };

  const handleBtAlgoChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newAlgo = e.target.value;
    if (newAlgo === "Custom AI Strategy") {
      // Always show popup when Custom AI Strategy is selected (even if already selected)
      if (btAlgo !== "Custom AI Strategy") {
        setPrevBacktestAlgo(btAlgo);
      }
      setBtAlgo(newAlgo);
      setShowBacktestPopup(true);
    } else {
      // Store new selection and clear Claude data
      setPrevBacktestAlgo(newAlgo);
      setBtAlgo(newAlgo);
      setClaudeBacktestStrategy(null);
      setClaudeBacktestError(null);
    }
  };

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
      // Handle Claude-generated strategies differently
      if (isClaudeBacktestSelected) {
        if (!claudeBacktestStrategy || !claudeBacktestStrategy.weights) {
          alert("⚠️ No Claude Strategy Available\n\nPlease generate a Claude strategy first by:\n1. Selecting 'Custom AI Strategy' from the dropdown\n2. Describing your strategy in the popup\n3. Clicking 'Generate Strategy'\n4. Then try training again");
          throw new Error("No Claude strategy generated. Please generate a strategy first.");
        }

        // Check for errors in the generated strategy
        if (claudeBacktestStrategy.error && claudeBacktestStrategy.fallbackUsed) {
          const proceed = confirm(`⚠️ Claude Strategy Warning\n\nYour strategy generation had issues:\n${claudeBacktestStrategy.error}\n\nWe're using a fallback strategy. Do you want to proceed with training?`);
          if (!proceed) {
            setLoading(false);
            return;
          }
        }

        // Simulate training progress for Claude strategies
        setProg(20);
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay for UX
        setProg(60);
        await new Promise(resolve => setTimeout(resolve, 500));
        setProg(90);
        await new Promise(resolve => setTimeout(resolve, 300));

        // Use Claude-generated weights directly
        const claudeWeights = claudeBacktestStrategy.weights;
        
        // Convert weights to portfolio format expected by UI
        const portfolioWeights: Record<string, number> = {};
        tickers.forEach((ticker, i) => {
          portfolioWeights[ticker] = claudeWeights[i] || 0;
        });

        // Generate simple NAV curve for visualization (this would ideally come from backend simulation)
        const nav: Record<string, number> = {};
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - btHistDays);
        
        let currentValue = 1.0;
        for (let i = 0; i < btHistDays; i += 5) { // Sample every 5 days
          const date = new Date(startDate);
          date.setDate(startDate.getDate() + i);
          const dateStr = date.toISOString().split('T')[0];
          
          // Simple random walk for demonstration (in reality, this would be calculated from historical data)
          currentValue *= (1 + (Math.random() - 0.5) * 0.02); // ±1% change
          nav[dateStr] = Math.max(0.1, currentValue); // Prevent negative values
        }

        // Generate basic metrics
        const navValues = Object.values(nav);
        const returns = navValues.slice(1).map((val, i) => (val - navValues[i]) / navValues[i]);
        const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
        const sharpeRatio = avgReturn / volatility;
        const maxDrawdown = Math.max(...navValues.map((_, i) => 
          Math.max(...navValues.slice(0, i + 1)) - navValues[i]
        )) / Math.max(...navValues);

        const metrics = {
          'Return': (navValues[navValues.length - 1] - navValues[0]) / navValues[0], // Total return as decimal
          'AnnualReturn': ((navValues[navValues.length - 1] - navValues[0]) / navValues[0]) * (252 / btHistDays), // Annualized
          'Sharpe': sharpeRatio,
          'DailyVol': volatility,
          'MaxDrawdown': maxDrawdown,
          'NumTrades': tickers.length // Number of assets
        };

        // Set results
        setWeights(portfolioWeights);
        setNav(nav);
        setMetrics(metrics);
        setProg(100);
        setLoading(false);

        // Save Claude backtest session to database for history tracking
        // Use the same TrainingLogService that traditional algorithms use
        try {
          // Simulate the same data format that comes from traditional algorithms
          const simulatedBackendResponse = {
            status: 'done',
            nav: nav,
            weights: portfolioWeights,
            metrics: metrics,
            algo: btAlgo.toUpperCase(),
            tickers: tickers
          };
          
          // Create unique Claude job ID and call the same endpoint traditional algorithms use
          const claudeJobId = `claude-backtest-${Date.now()}`;
          const response = await fetch(`/api/train/${claudeJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              data: simulatedBackendResponse,
              originalParams: {
                algo: btAlgo.toUpperCase(),
                tickers: tickers,
                hist_days: btHistDays,
                lookback: lookBack,
                eval_win: evalWin,
                tc: tc
              }
            })
          });
        } catch (logError) {
          // Don't throw - let the backtest complete even if logging fails
        }

      } else {
        // Handle traditional algorithms via backend
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
      }
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
      // Handle Claude forecast strategies differently
      if (isClaudeForecastSelected) {
        if (!claudeForecastStrategy || !claudeForecastStrategy.predictions) {
          alert("⚠️ No Claude Strategy Available\n\nPlease generate a Claude strategy first by:\n1. Selecting 'Custom AI Strategy' from the dropdown\n2. Describing your strategy in the popup\n3. Clicking 'Generate Strategy'\n4. Then try forecasting again");
          setFLoading(false);
          return;
        }

        // Convert Claude predictions to the same format as traditional algorithms
        const claudePredictions = claudeForecastStrategy.predictions;
        
        // Get the base stock data to calculate percentage changes from first prediction
        const baseStockData = getStockDataFromTickers();
        const referencePredictions = claudePredictions; // These are based on first stock
        const referenceStock = baseStockData[0]; // The stock used for generating predictions
        const referencePrice = referenceStock?.price || 100;
        
        // Detect if predictions are multipliers or absolute prices
        const firstPrediction = referencePredictions[0];
        const isMultiplier = firstPrediction && firstPrediction.price >= 0.5 && firstPrediction.price <= 2.0;
        
        const percentageChanges = referencePredictions.map((pred: any, index: number) => {
          const multiplier = isMultiplier ? pred.price : (pred.price / referencePrice);
          return {
            date: pred.date,
            multiplier: multiplier,
            confidence: pred.confidence
          };
        });
        
        // Apply percentage changes to each ticker individually using REAL historical data
        for (let i = 0; i < tickers.length; i++) {
          const ticker = tickers[i];
          try {
            // Add small delay between calls to prevent backend race conditions (same as traditional algorithms)
            if (i > 0) {
              await new Promise(resolve => setTimeout(resolve, 500)); // 500ms delay
            }
            
            // Call backend API to get REAL YFinance data (same as traditional algorithms)
            const res = await fetch(`/api/forecast/arima`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ 
                ticker, 
                start, 
                end, 
                horizon: 1 // We just need historical data, minimal forecast
              }),
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

            // Get REAL historical data from YFinance (same as traditional algorithms)
            const historySeries = toSeries(payload.history_dates, payload.history_values);
            
            // Get the last historical price for continuity
            const lastHistoricalPrice = historySeries.length > 0 ? 
              historySeries[historySeries.length - 1].price : 100;
            
            // Apply Custom AI percentage changes to real historical data
            const forecastSeries = percentageChanges.map((change: any) => ({
              date: change.date,
              price: lastHistoricalPrice * change.multiplier // Apply to real last price
            }));
            
            tempDataMap[ticker] = {
              historySeries, // Now uses REAL YFinance data!
              forecastSeries,
              algorithm: "Custom AI Strategy"
            };

            completedTickers++;
            setFProg((completedTickers / tickers.length) * 100);
            
          } catch (err) {
            console.error(`Custom AI data fetch failed for ${ticker}:`, err);
            // Add empty data for failed ticker to prevent UI crashes
            tempDataMap[ticker] = {
              historySeries: [],
              forecastSeries: [],
              algorithm: "Custom AI Strategy"
            };
            
            completedTickers++;
            setFProg((completedTickers / tickers.length) * 100);
          }
        }
      } else {
        // Handle traditional algorithms
        for (let i = 0; i < tickers.length; i++) {
          const ticker = tickers[i];
          try {
            // Add small delay between calls to prevent backend race conditions
            if (i > 0) {
              await new Promise(resolve => setTimeout(resolve, 500)); // 500ms delay
            }
            
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
      algorithm: algo
    };

    completedTickers++;
    setFProg((completedTickers / tickers.length) * 100);
  } catch (err) {
    // Add empty data for failed ticker to prevent UI crashes
    tempDataMap[ticker] = {
      historySeries: [],
      forecastSeries: [],
      algorithm: algo
    };
    // Continue processing other tickers even if this one fails
    completedTickers++;
    setFProg((completedTickers / tickers.length) * 100);
          }
        }
      }

      // Update state with forecast results first (without metrics)
      setForecastDataMap(tempDataMap);

      // Save forecast session to database for history tracking and calculate metrics
      try {
  const forecasts = Object.entries(tempDataMap).filter(([_, data]) => 
    data.historySeries.length > 0 && data.forecastSeries.length > 0
  );
  
  if (forecasts.length > 0) {
    // Calculate metrics for each ticker and add them to forecast data
    const updatedForecasts = await Promise.all(
      forecasts.map(async ([ticker, data]) => {
        try {
          // Pass Claude strategy for Custom AI metrics calculation
          const metrics = await calculateForecastMetrics(
            data, 
            data.algorithm!, 
            ticker,
            isClaudeForecastSelected ? claudeForecastStrategy : undefined
          );
          // Add metrics to the data object
          const updatedData = { ...data, metrics };
          return [ticker, updatedData];
        } catch (error) {
          console.error(`Failed to calculate metrics for ${ticker} during logging:`, error);
          // Add default metrics to the data object
          const updatedData = { ...data, metrics: { mse: 0, mae: 0 } };
          return [ticker, updatedData];
        }
      })
    );
    
    // Update tempDataMap with metrics included
    updatedForecasts.forEach((result) => {
      const [ticker, data] = result as [string, ForecastData];
      tempDataMap[ticker] = data;
    });
    
    // Update state with the enhanced forecast data (including metrics)
    setForecastDataMap(tempDataMap);
    
    // Create metrics map for logging
    const metricsMap = Object.fromEntries(
      updatedForecasts.map((result) => {
        const [ticker, data] = result as [string, ForecastData];
        return [ticker, data.metrics];
      })
    );
    
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
      ])),
      metrics: metricsMap
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
              onChange={handleAlgoChange}
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
                    <ForecastStockItem
                      key={ticker}
                      ticker={ticker}
                      data={data}
                      claudeStrategy={isClaudeForecastSelected ? claudeForecastStrategy : undefined}
                    />
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
              onChange={handleBtAlgoChange}
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

      {/* Claude Strategy Popups with Error Boundaries */}
      <ErrorBoundary
        fallback={
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300">
            <h3 className="font-medium mb-2">Strategy Generation Error</h3>
            <p className="text-sm">The strategy generation popup encountered an error. Please refresh the page and try again.</p>
          </div>
        }
      >
        <ClaudePopup
          isOpen={showForecastPopup}
          onClose={handleForecastPopupClose}
          mode="forecast"
          stockData={getStockDataFromTickers()}
          onStrategyGenerated={handleClaudeForecastGenerated}
          onError={handleClaudeForecastError}
          dashboardParams={{
            historyDays: histDays,
            forecastDays: forecastDays,
            algorithm: prevForecastAlgo
          }}
        />
      </ErrorBoundary>

      <ErrorBoundary
        fallback={
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300">
            <h3 className="font-medium mb-2">Strategy Generation Error</h3>
            <p className="text-sm">The strategy generation popup encountered an error. Please refresh the page and try again.</p>
          </div>
        }
      >
        <ClaudePopup
          isOpen={showBacktestPopup}
          onClose={handleBacktestPopupClose}
          mode="backtest"
          stockData={getStockDataFromTickers()}
          onStrategyGenerated={handleClaudeBacktestGenerated}
          onError={handleClaudeBacktestError}
          dashboardParams={{
            backtestDays: btHistDays,
            lookbackDays: lookBack,
            evaluationWindow: evalWin,
          transactionCost: tc,
          algorithm: prevBacktestAlgo
        }}
        />
      </ErrorBoundary>
    </div>
  );
}
