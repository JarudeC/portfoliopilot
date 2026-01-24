/**
 * Forecast hook for stock price prediction.
 * Manages state, execution flow, and Claude AI strategy integration for forecasting.
 */
import { useState, useCallback } from 'react';
import { useToast } from '../../../components/ui/Toast';
import { type GenerationResult, executeUserCode } from '../../../lib/claude/client';
import {
  calculateForecastMetrics,
  calculateOverallForecastMetrics,
  type ForecastData,
  type OverallForecastMetrics
} from '../../../lib/utils/forecastMetrics';

export const FORECAST_ALGOS = ["ARIMA", "LSTM", "Autoformer", "Custom AI Strategy"];
export const HIST_DAYS = [60, 90, 180, 365];
export const FORECAST_DAYS = [5, 7, 14, 30];

export interface ForecastParams {
  histDays: number;
  forecastDays: number;
}

export interface UseForecastReturn {
  algo: string;
  setAlgo: (algo: string) => void;
  params: ForecastParams;
  setParams: React.Dispatch<React.SetStateAction<ForecastParams>>;
  loading: boolean;
  progress: number;
  forecastDataMap: Record<string, ForecastData>;
  forecastingTickers: string[];
  overallMetrics: OverallForecastMetrics | null;
  overallMetricsLoading: boolean;
  claudeStrategy: GenerationResult | null;
  showPopup: boolean;
  openPopup: () => void;
  runForecast: (tickers: string[]) => Promise<void>;
  handleAlgoChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  handleClaudeGenerated: (result: GenerationResult) => void;
  handleClaudeError: (error: Error) => void;
  handlePopupClose: () => void;
  resetParams: () => void;
}

const DEFAULT_PARAMS: ForecastParams = {
  histDays: 180,
  forecastDays: 14,
};

export function useForecast(): UseForecastReturn {
  const { showSuccess, showWarning, showError } = useToast();

  const [algo, setAlgo] = useState(FORECAST_ALGOS[0]);
  const [prevAlgo, setPrevAlgo] = useState(FORECAST_ALGOS[0]);
  const [params, setParams] = useState<ForecastParams>(DEFAULT_PARAMS);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [forecastDataMap, setForecastDataMap] = useState<Record<string, ForecastData>>({});
  const [forecastingTickers, setForecastingTickers] = useState<string[]>([]);
  const [overallMetrics, setOverallMetrics] = useState<OverallForecastMetrics | null>(null);
  const [overallMetricsLoading, setOverallMetricsLoading] = useState(false);
  const [claudeStrategy, setClaudeStrategy] = useState<GenerationResult | null>(null);
  const [showPopup, setShowPopup] = useState(false);

  const isClaudeSelected = algo === "Custom AI Strategy";

  /**
   * Handle algorithm selection change. Opens Claude popup for custom strategies.
   */
  const handleAlgoChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newAlgo = e.target.value;
    if (newAlgo === "Custom AI Strategy") {
      if (algo !== "Custom AI Strategy") {
        setPrevAlgo(algo);
      }
      setAlgo(newAlgo);
      setShowPopup(true);
    } else {
      setPrevAlgo(newAlgo);
      setAlgo(newAlgo);
      setClaudeStrategy(null);
    }
  }, [algo]);

  /**
   * Handle successful Claude strategy generation.
   */
  const handleClaudeGenerated = useCallback((result: GenerationResult) => {
    setClaudeStrategy(result);
    setShowPopup(false);

    if (result.fallbackUsed) {
      showWarning(
        "Forecast Strategy Generated (Fallback)",
        `Strategy generated using fallback method: ${result.error || 'AI generation failed'}`
      );
    } else if (result.loadedFromSaved) {
      showSuccess(
        "Forecast Strategy Loaded",
        "Your saved forecast strategy has been loaded and is ready to use."
      );
    } else {
      showSuccess(
        "Forecast Strategy Generated",
        "Your custom AI forecast strategy has been successfully created and is ready to use."
      );
    }
  }, [showSuccess, showWarning]);

  /**
   * Handle Claude strategy generation error.
   */
  const handleClaudeError = useCallback((error: Error) => {
    setClaudeStrategy(null);
    setShowPopup(false);
    setAlgo(prevAlgo);

    showError(
      "Forecast Strategy Failed",
      error.message,
      ["Try again with a simpler description", "Check your internet connection", "Use more common technical analysis terms"]
    );
  }, [prevAlgo, showError]);

  /**
   * Handle popup close. Reverts algorithm selection if no strategy was generated.
   */
  const handlePopupClose = useCallback(() => {
    setShowPopup(false);
    if (!claudeStrategy) {
      setAlgo(prevAlgo);
    }
  }, [claudeStrategy, prevAlgo]);

  /**
   * Open the Claude popup manually (for reconfiguring existing strategy).
   */
  const openPopup = useCallback(() => {
    setShowPopup(true);
  }, []);

  const resetParams = useCallback(() => {
    setParams(DEFAULT_PARAMS);
  }, []);

  /**
   * Execute forecast for selected tickers.
   * Routes to Claude AI execution or traditional backend algorithms.
   */
  const runForecast = useCallback(async (tickers: string[]) => {
    if (!tickers.length || loading) return;

    // Capture current values to avoid stale closures in inner functions
    const currentParams = { ...params };
    const currentAlgo = algo;
    const currentStrategy = claudeStrategy;

    setLoading(true);
    setProgress(0);
    setForecastDataMap({});
    setOverallMetrics(null);
    setForecastingTickers([...tickers]);

    const today = new Date();
    const end = today.toISOString().slice(0, 10);
    const tradingDayMultiplier = 1.43;
    const calendarDaysBack = Math.round(currentParams.histDays * tradingDayMultiplier);
    const start = new Date(today.getTime() - calendarDaysBack * 86_400_000).toISOString().slice(0, 10);

    const totalTickers = tickers.length;
    let completedTickers = 0;
    const tempDataMap: Record<string, ForecastData> = {};

    try {
      if (isClaudeSelected) {
        await runClaudeForecast(tickers, start, end, tempDataMap, currentParams, currentStrategy, () => {
          completedTickers++;
          setProgress((completedTickers / totalTickers) * 100);
        });
      } else {
        await runTraditionalForecast(tickers, start, end, tempDataMap, currentParams, currentAlgo, () => {
          completedTickers++;
          setProgress((completedTickers / totalTickers) * 100);
        });
      }

      setForecastDataMap(tempDataMap);
      await calculateAndLogMetrics(tempDataMap, start, end, currentParams, currentAlgo, currentStrategy);
      setLoading(false);

    } catch (e) {
      console.error(e);
      alert((e as Error).message);
      setLoading(false);
    }
  }, [loading, isClaudeSelected, params, algo, claudeStrategy]);

  /**
   * Execute Claude AI strategy forecast with real data.
   * Fetches real prices from /api/prices and executes the AI-generated code.
   */
  async function runClaudeForecast(
    tickers: string[],
    start: string,
    end: string,
    tempDataMap: Record<string, ForecastData>,
    currentParams: ForecastParams,
    currentStrategy: GenerationResult | null,
    onProgress: () => void
  ) {
    if (!currentStrategy || !currentStrategy.code) {
      alert("⚠️ No Claude Strategy Available\n\nPlease generate a Claude strategy first.");
      setLoading(false);
      return;
    }

    for (let i = 0; i < tickers.length; i++) {
      const ticker = tickers[i];
      try {
        if (i > 0) await new Promise(resolve => setTimeout(resolve, 500));

        // Fetch real historical prices
        const res = await fetch(`/api/prices`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, start, end }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (!payload.dates || !payload.prices) throw new Error(`Invalid response for ${ticker}`);

        const historySeries = payload.dates.map((date: string, index: number) => ({
          date,
          price: payload.prices[index]
        }));

        // Build stockData with real prices for AI code execution
        const stockData = [{
          symbol: ticker,
          price: payload.prices[payload.prices.length - 1],
          lookbackPrices: payload.prices,
          lookbackDates: payload.dates
        }];

        // Execute the AI-generated code with real data
        const result = await executeUserCode(
          currentStrategy.code,
          'forecast',
          stockData,
          undefined, // securityConfig
          currentParams.forecastDays,
          {
            historyDays: currentParams.histDays
          }
        );

        let forecastSeries = [];
        if (result.success && result.predictions && Array.isArray(result.predictions)) {
          const lastPrice = payload.prices[payload.prices.length - 1] || 100;
          const firstPred = result.predictions[0];
          const isMultiplier = firstPred && firstPred.price >= 0.5 && firstPred.price <= 2.0;

          forecastSeries = result.predictions.map((pred: any) => ({
            date: pred.date,
            price: isMultiplier ? lastPrice * pred.price : pred.price
          }));
        } else {
          // Fallback if execution failed
          forecastSeries = generateFallbackForecast(payload.prices, end, currentParams.forecastDays);
        }

        tempDataMap[ticker] = { historySeries, forecastSeries, algorithm: "Custom AI Strategy" };
        onProgress();

      } catch (err) {
        console.error(`Custom AI data fetch failed for ${ticker}:`, err);
        tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: "Custom AI Strategy" };
        onProgress();
      }
    }
  }

  /**
   * Execute traditional algorithm forecast via backend API.
   */
  async function runTraditionalForecast(
    tickers: string[],
    start: string,
    end: string,
    tempDataMap: Record<string, ForecastData>,
    currentParams: ForecastParams,
    currentAlgo: string,
    onProgress: () => void
  ) {
    for (let i = 0; i < tickers.length; i++) {
      const ticker = tickers[i];
      try {
        if (i > 0) await new Promise(resolve => setTimeout(resolve, 500));

        const res = await fetch(`/api/forecast/${currentAlgo.toLowerCase()}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, start, end, horizon: currentParams.forecastDays }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();

        if (!payload.history_dates || !payload.history_values ||
            !payload.forecast_dates || !payload.forecast_values) {
          throw new Error(`Invalid response structure for ${ticker}`);
        }

        const toSeries = (d: string[], v: number[]) =>
          d.map((x, i) => ({ date: x, price: v[i] }));

        tempDataMap[ticker] = {
          historySeries: toSeries(payload.history_dates, payload.history_values),
          forecastSeries: toSeries(payload.forecast_dates, payload.forecast_values),
          algorithm: currentAlgo
        };

        onProgress();

      } catch (err) {
        console.error(`Forecast failed for ${ticker}:`, err);
        tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: currentAlgo };
        onProgress();
      }
    }
  }

  /**
   * Generate fallback forecast when Claude predictions are unavailable.
   */
  function generateFallbackForecast(prices: number[], end: string, forecastDays: number) {
    const lastPrice = prices[prices.length - 1] || 100;
    const startDate = new Date(end);
    const forecastSeries = [];

    for (let j = 1; j <= forecastDays; j++) {
      const futureDate = new Date(startDate);
      futureDate.setDate(startDate.getDate() + j);
      forecastSeries.push({
        date: futureDate.toISOString().split('T')[0],
        price: lastPrice * (1 + (Math.random() - 0.5) * 0.02)
      });
    }

    return forecastSeries;
  }

  /**
   * Calculate overall metrics and log forecast results for persistence.
   */
  async function calculateAndLogMetrics(
    tempDataMap: Record<string, ForecastData>,
    start: string,
    end: string,
    currentParams: ForecastParams,
    currentAlgo: string,
    currentStrategy: GenerationResult | null
  ) {
    const forecasts = Object.entries(tempDataMap).filter(([_, data]) =>
      data.historySeries.length > 0 && data.forecastSeries.length > 0
    );

    if (forecasts.length === 0) return;

    const currentIsClaudeSelected = currentAlgo === "Custom AI Strategy";

    const updatedForecasts = await Promise.all(
      forecasts.map(async ([ticker, data]) => {
        try {
          const metrics = await calculateForecastMetrics(
            data,
            data.algorithm!,
            ticker,
            currentIsClaudeSelected ? currentStrategy : undefined
          );
          return [ticker, { ...data, metrics }];
        } catch (error) {
          console.error(`Failed to calculate metrics for ${ticker}:`, error);
          return [ticker, { ...data, metrics: { mse: 0, mae: 0 } }];
        }
      })
    );

    updatedForecasts.forEach((result) => {
      const [ticker, data] = result as [string, ForecastData];
      tempDataMap[ticker] = data;
    });

    setForecastDataMap({ ...tempDataMap });

    let metricsForLogging = null;
    try {
      setOverallMetricsLoading(true);
      const stockDataList = updatedForecasts.map((result) => {
        const [ticker, data] = result as [string, ForecastData];
        return {
          ticker,
          data,
          forecastAlgorithm: data.algorithm!,
          claudeStrategy: currentIsClaudeSelected ? currentStrategy : undefined
        };
      });

      const overallMetricsResult = await calculateOverallForecastMetrics(stockDataList);
      setOverallMetrics(overallMetricsResult);

      metricsForLogging = overallMetricsResult ? {
        mse: overallMetricsResult.mse,
        mae: overallMetricsResult.mae,
        total_predictions: overallMetricsResult.totalPredictions,
        stock_count: overallMetricsResult.stockCount
      } : null;

    } catch (error) {
      console.error('Failed to calculate overall metrics:', error);
      setOverallMetrics(null);
    } finally {
      setOverallMetricsLoading(false);
    }

    await logForecastResult(forecasts, start, end, metricsForLogging, currentParams, currentAlgo);
  }

  /**
   * Log forecast results to history endpoint for persistence.
   */
  async function logForecastResult(
    forecasts: [string, ForecastData][],
    start: string,
    end: string,
    metricsForLogging: any,
    currentParams: ForecastParams,
    currentAlgo: string
  ) {
    try {
      const logPayload = {
        type: 'forecast',
        stocks: forecasts.map(([ticker]) => ticker),
        model: currentAlgo.toUpperCase(),
        parameters: {
          algorithm: currentAlgo,
          start_date: start,
          end_date: end,
          history_days: currentParams.histDays,
          forecast_days: currentParams.forecastDays,
          tickers: forecasts.map(([ticker]) => ticker)
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
        metrics: metricsForLogging
      };

      await fetch('/api/forecast/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logPayload)
      });
    } catch (logError) {
      console.error('Failed to log forecast:', logError);
    }
  }

  return {
    algo,
    setAlgo,
    params,
    setParams,
    loading,
    progress,
    forecastDataMap,
    forecastingTickers,
    overallMetrics,
    overallMetricsLoading,
    claudeStrategy,
    showPopup,
    openPopup,
    runForecast,
    handleAlgoChange,
    handleClaudeGenerated,
    handleClaudeError,
    handlePopupClose,
    resetParams,
  };
}
