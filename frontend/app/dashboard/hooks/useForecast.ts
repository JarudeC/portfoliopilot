/**
 * Forecast hook for stock price prediction.
 * Manages state, execution flow, and Claude AI strategy integration for forecasting.
 */
import { useState, useCallback } from 'react';
import { useToast } from '../../../components/ui/Toast';
import { type GenerationResult, executeUserCode } from '../../../lib/claude/client';
import {
  calculateForecastMetrics,
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
   * Fetches all prices in one batch request, then executes AI code for each ticker.
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

    // Fetch all prices in one batch request
    const batchRes = await fetch(`/api/prices/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers, start, end }),
    });

    if (!batchRes.ok) {
      throw new Error(`Batch price fetch failed: HTTP ${batchRes.status}`);
    }

    const batchData = await batchRes.json();

    // Process each ticker with the fetched data
    for (const ticker of tickers) {
      try {
        const payload = batchData[ticker];

        if (!payload || payload.error || !payload.dates || !payload.prices) {
          console.error(`Failed to fetch data for ${ticker}:`, payload?.error);
          tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: "Custom AI Strategy" };
          onProgress();
          continue;
        }

        const historySeries = payload.dates.map((date: string, index: number) => ({
          date,
          price: payload.prices[index]
        }));

        const stockData = [{
          symbol: ticker,
          price: payload.prices[payload.prices.length - 1],
          lookbackPrices: payload.prices,
          lookbackDates: payload.dates
        }];

        const result = await executeUserCode(
          currentStrategy.code,
          'forecast',
          stockData,
          undefined,
          currentParams.forecastDays,
          { historyDays: currentParams.histDays }
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
          forecastSeries = generateFallbackForecast(payload.prices, end, currentParams.forecastDays);
        }

        tempDataMap[ticker] = { historySeries, forecastSeries, algorithm: "Custom AI Strategy" };
        onProgress();

      } catch (err) {
        console.error(`Custom AI processing failed for ${ticker}:`, err);
        tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: "Custom AI Strategy" };
        onProgress();
      }
    }
  }

  /**
   * Execute traditional algorithm forecast via batch backend API.
   * Fetches all prices in parallel on backend, then runs models sequentially.
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
    const toSeries = (d: string[], v: number[]) =>
      d.map((x, i) => ({ date: x, price: v[i] }));

    try {
      const res = await fetch(`/api/forecast/${currentAlgo.toLowerCase()}/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers,
          start,
          end,
          horizon: currentParams.forecastDays,
          calculate_metrics: true  // Request backend to calculate MSE/MAE
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const batchData = await res.json();

      for (const ticker of tickers) {
        const payload = batchData[ticker];

        if (!payload || payload.error) {
          console.error(`Forecast failed for ${ticker}:`, payload?.error);
          tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: currentAlgo };
        } else if (!payload.history_dates || !payload.forecast_dates) {
          console.error(`Invalid response structure for ${ticker}`);
          tempDataMap[ticker] = { historySeries: [], forecastSeries: [], algorithm: currentAlgo };
        } else {
          tempDataMap[ticker] = {
            historySeries: toSeries(payload.history_dates, payload.history_values),
            forecastSeries: toSeries(payload.forecast_dates, payload.forecast_values),
            algorithm: currentAlgo,
            metrics: payload.metrics  // Use backend-calculated metrics
          };
        }
        onProgress();
      }
    } catch (err) {
      console.error('Batch forecast failed:', err);
      for (const ticker of tickers) {
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
   * For classical algorithms, metrics come from backend. For Custom AI, calculate frontend-side.
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

    // For Custom AI, calculate metrics frontend-side. For classical, metrics already in data from backend.
    const updatedForecasts = await Promise.all(
      forecasts.map(async ([ticker, data]) => {
        // If metrics already present (from backend), use them
        if (data.metrics && (data.metrics.mse > 0 || data.metrics.mae > 0)) {
          return [ticker, data];
        }

        // Only calculate frontend-side for Custom AI
        if (currentIsClaudeSelected) {
          try {
            const metrics = await calculateForecastMetrics(
              data,
              data.algorithm!,
              ticker,
              currentStrategy
            );
            return [ticker, { ...data, metrics }];
          } catch (error) {
            console.error(`Failed to calculate metrics for ${ticker}:`, error);
            return [ticker, { ...data, metrics: { mse: 0, mae: 0 } }];
          }
        }

        // Classical with no metrics - shouldn't happen, but handle gracefully
        return [ticker, { ...data, metrics: { mse: 0, mae: 0 } }];
      })
    );

    updatedForecasts.forEach((result) => {
      const [ticker, data] = result as [string, ForecastData];
      tempDataMap[ticker] = data;
    });

    setForecastDataMap({ ...tempDataMap });

    // Calculate overall metrics by aggregating individual metrics
    let metricsForLogging = null;
    try {
      setOverallMetricsLoading(true);

      const allMetrics = updatedForecasts
        .map((result) => (result as [string, ForecastData])[1].metrics)
        .filter((m): m is { mse: number; mae: number } => m !== undefined && (m.mse > 0 || m.mae > 0));

      if (allMetrics.length > 0) {
        const overallMetricsResult = {
          mse: allMetrics.reduce((sum, m) => sum + m.mse, 0) / allMetrics.length,
          mae: allMetrics.reduce((sum, m) => sum + m.mae, 0) / allMetrics.length,
          totalPredictions: allMetrics.length,
          stockCount: allMetrics.length
        };
        setOverallMetrics(overallMetricsResult);

        metricsForLogging = {
          mse: overallMetricsResult.mse,
          mae: overallMetricsResult.mae,
          total_predictions: overallMetricsResult.totalPredictions,
          stock_count: overallMetricsResult.stockCount
        };
      } else {
        setOverallMetrics(null);
      }
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
