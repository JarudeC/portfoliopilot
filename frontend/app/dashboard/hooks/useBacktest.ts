/**
 * Backtest hook for portfolio optimization strategies.
 * Manages state, execution flow, and Claude AI strategy integration for backtesting.
 */
import { useState, useCallback } from 'react';
import { useToast } from '../../../components/ui/Toast';
import { type GenerationResult } from '../../../lib/claude/client';

export const BACKTEST_ALGOS = ["Naive Markowitz", "GVMP", "PPN", "Margin Trader", "Custom AI Strategy"];
export const LOOKBACKS = [30, 60, 90];
export const EVALWINS = [5, 10, 15];
export const BTHIST_DAYS = [365, 1095, 1825];

export interface BacktestParams {
  btHistDays: number;
  /**
   * Lookback window (in trading days) used to estimate covariances/trends for weight calculation.
   * The algorithm analyzes this many days of historical data before each rebalancing decision.
   * Example: lookBack=30 means use 30 days of price history to compute optimal weights.
   */
  lookBack: number;
  /**
   * Evaluation window (in trading days) - how often to rebalance the portfolio.
   * After calculating weights, hold them for this many days before recalculating.
   * Example: evalWin=5 means rebalance every 5 trading days (~1 week).
   */
  evalWin: number;
  tc: number;
}

export interface BacktestResults {
  nav: Record<string, number> | null;
  weights: Record<string, number> | null;
  metrics: Record<string, string | number> | null;
  usedParams: BacktestParams | null;
}

export interface UseBacktestReturn {
  btAlgo: string;
  setBtAlgo: (algo: string) => void;
  params: BacktestParams;
  setParams: React.Dispatch<React.SetStateAction<BacktestParams>>;
  loading: boolean;
  progress: number;
  results: BacktestResults;
  claudeStrategy: GenerationResult | null;
  showPopup: boolean;
  openPopup: () => void;
  runBacktest: (tickers: string[]) => Promise<void>;
  handleAlgoChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  handleClaudeGenerated: (result: GenerationResult) => void;
  handleClaudeError: (error: Error) => void;
  handlePopupClose: () => void;
  resetParams: () => void;
}

const DEFAULT_PARAMS: BacktestParams = {
  btHistDays: 365,
  lookBack: 30,
  evalWin: 5,
  tc: 0.002,
};

export function useBacktest(): UseBacktestReturn {
  const { showSuccess, showWarning, showError } = useToast();

  const [btAlgo, setBtAlgo] = useState(BACKTEST_ALGOS[0]);
  const [prevAlgo, setPrevAlgo] = useState(BACKTEST_ALGOS[0]);
  const [params, setParams] = useState<BacktestParams>(DEFAULT_PARAMS);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<BacktestResults>({
    nav: null,
    weights: null,
    metrics: null,
    usedParams: null,
  });
  const [claudeStrategy, setClaudeStrategy] = useState<GenerationResult | null>(null);
  const [showPopup, setShowPopup] = useState(false);

  const isClaudeSelected = btAlgo === "Custom AI Strategy";

  /**
   * Handle algorithm selection change. Opens Claude popup for custom strategies.
   */
  const handleAlgoChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newAlgo = e.target.value;
    if (newAlgo === "Custom AI Strategy") {
      if (btAlgo !== "Custom AI Strategy") {
        setPrevAlgo(btAlgo);
      }
      setBtAlgo(newAlgo);
      setShowPopup(true);
    } else {
      setPrevAlgo(newAlgo);
      setBtAlgo(newAlgo);
      setClaudeStrategy(null);
    }
  }, [btAlgo]);

  /**
   * Handle successful Claude strategy generation.
   */
  const handleClaudeGenerated = useCallback((result: GenerationResult) => {
    setClaudeStrategy(result);
    setShowPopup(false);

    if (result.fallbackUsed) {
      showWarning(
        "Backtest Strategy Generated (Fallback)",
        `Strategy generated using fallback method: ${result.error || 'AI generation failed'}`
      );
    } else if (result.loadedFromSaved) {
      showSuccess(
        "Backtest Strategy Loaded",
        "Your saved backtest strategy has been loaded and is ready to use."
      );
    } else {
      showSuccess(
        "Backtest Strategy Generated",
        "Your custom AI backtest strategy has been successfully created and is ready to use."
      );
    }
  }, [showSuccess, showWarning]);

  /**
   * Handle Claude strategy generation error.
   */
  const handleClaudeError = useCallback((error: Error) => {
    setClaudeStrategy(null);
    setShowPopup(false);
    setBtAlgo(prevAlgo);

    showError(
      "Backtest Strategy Failed",
      error.message,
      ["Try again with a simpler description", "Check your internet connection", "Use more common investment terms"]
    );
  }, [prevAlgo, showError]);

  /**
   * Handle popup close. Reverts algorithm selection if no strategy was generated.
   */
  const handlePopupClose = useCallback(() => {
    setShowPopup(false);
    if (!claudeStrategy) {
      setBtAlgo(prevAlgo);
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
   * Execute backtest for selected tickers.
   * Routes to Claude AI execution or traditional backend algorithms.
   */
  const runBacktest = useCallback(async (tickers: string[]) => {
    if (!tickers.length || loading) return;

    // Capture current values to avoid stale closures in inner functions
    const currentParams = { ...params };
    const currentAlgo = btAlgo;
    const currentStrategy = claudeStrategy;

    setLoading(true);
    setProgress(5);
    setResults({ nav: null, weights: null, metrics: null, usedParams: null });

    try {
      if (isClaudeSelected) {
        await runClaudeBacktest(tickers, currentParams, currentAlgo, currentStrategy);
      } else {
        await runTraditionalBacktest(tickers, currentParams, currentAlgo);
      }
    } catch (e) {
      console.error(e);
      alert((e as Error).message);
      setLoading(false);
    }
  }, [loading, isClaudeSelected, claudeStrategy, params, btAlgo]);

  /**
   * Execute Claude AI strategy backtest with dynamic rebalancing.
   */
  async function runClaudeBacktest(
    tickers: string[],
    currentParams: BacktestParams,
    currentAlgo: string,
    currentStrategy: GenerationResult | null
  ) {
    if (!currentStrategy || !currentStrategy.code) {
      alert("⚠️ No Claude Strategy Available\n\nPlease generate a Claude strategy first.");
      throw new Error("No Claude strategy generated.");
    }

    if (currentStrategy.error && currentStrategy.fallbackUsed) {
      const proceed = confirm(`⚠️ Claude Strategy Warning\n\n${currentStrategy.error}\n\nProceed with fallback?`);
      if (!proceed) {
        setLoading(false);
        return;
      }
    }

    setProgress(20);
    await new Promise(resolve => setTimeout(resolve, 500));
    setProgress(60);
    await new Promise(resolve => setTimeout(resolve, 500));
    setProgress(90);

    // Initialize with equal weights - actual weights computed dynamically by AI code during backtest
    const defaultWeight = 1.0 / tickers.length;
    let portfolioWeights: Record<string, number> = {};
    tickers.forEach((ticker) => {
      portfolioWeights[ticker] = defaultWeight;
    });

    const nav: Record<string, number> = {};
    let metrics: Record<string, string | number> = {};

    try {
      const today = new Date();
      const end = today.toISOString().slice(0, 10);
      const tradingDayMultiplier = 1.43;
      const calendarDaysBack = Math.round(currentParams.btHistDays * tradingDayMultiplier);
      const start = new Date(today.getTime() - calendarDaysBack * 86_400_000).toISOString().slice(0, 10);

      // Fetch all prices in one batch request
      const batchRes = await fetch(`/api/prices/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers, start, end }),
      });

      if (!batchRes.ok) throw new Error(`Batch price fetch failed: HTTP ${batchRes.status}`);
      const batchData = await batchRes.json();

      const allTickerData = tickers.map((ticker) => {
        const payload = batchData[ticker];
        if (!payload || payload.error || !payload.dates || !payload.prices) {
          throw new Error(`Invalid response for ${ticker}: ${payload?.error || 'missing data'}`);
        }
        return {
          ticker,
          weight: defaultWeight,
          dates: payload.dates,
          prices: payload.prices
        };
      });

      if (allTickerData.length > 0 && allTickerData[0].dates.length > 1) {
        const calculationResult = await calculatePortfolioReturns(allTickerData, tickers, portfolioWeights, currentParams, currentStrategy);
        Object.assign(nav, calculationResult.nav);
        metrics = calculationResult.metrics;
        portfolioWeights = calculationResult.portfolioWeights;
      } else {
        throw new Error('Insufficient historical data');
      }

    } catch (error) {
      console.error('Custom AI backtest failed:', error);
      const today = new Date().toISOString().split('T')[0];
      nav[today] = 1.0;
      metrics = {
        'Return': "0.00%", 'AnnualReturn': "0.00%", 'DailyVol': "0.00%",
        'AnnualVol': "0.00%", 'Sharpe': "0.00", 'Sortino': "0.00"
      };
    }

    setResults({ nav, weights: portfolioWeights, metrics, usedParams: currentParams });
    setProgress(100);
    setLoading(false);

    await logBacktestResult(nav, portfolioWeights, metrics, tickers, currentParams, currentAlgo);
  }

  /**
   * Calculate portfolio returns with dynamic AI-driven rebalancing.
   *
   * REBALANCING SIMULATION:
   * The backtest simulates a trading strategy where the portfolio is rebalanced every `evalWin` days.
   * At each rebalancing point, the algorithm uses `lookBack` days of historical data to calculate
   * optimal portfolio weights, then holds those weights until the next rebalancing.
   *
   * Timeline example (lookBack=30, evalWin=5):
   *   Days 0-29:  [lookBack period - used to estimate initial weights]
   *   Day 30:     Calculate weights using days 0-29, then BUY according to weights
   *   Days 30-34: HOLD - track daily returns using day-30 weights
   *   Day 35:     Recalculate weights using days 5-34, then REBALANCE (sell/buy to match new weights)
   *   Days 35-39: HOLD - track daily returns using day-35 weights
   *   ... repeat until end of backtest period
   */
  async function calculatePortfolioReturns(
    allTickerData: Array<{ ticker: string; weight: number; dates: string[]; prices: number[] }>,
    tickers: string[],
    portfolioWeights: Record<string, number>,
    currentParams: BacktestParams,
    currentStrategy: GenerationResult | null
  ) {
    const dates = allTickerData[0].dates;
    const prices = allTickerData.map(t => t.prices);
    const nav: Record<string, number> = {};

    // firstSig: First day we can generate a trading signal (need lookBack days of history first)
    const firstSig = currentParams.lookBack;
    // numWin: Total number of rebalancing windows in the backtest period
    const numWin = Math.floor((dates.length - firstSig) / currentParams.evalWin);

    let portfolioValue = 1.0;
    let prevWeights: number[] | null = null;
    let finalWeights: number[] | null = null;
    const portfolioReturns: number[] = [];

    // Loop through each rebalancing window (each "step" = one evalWin period)
    for (let step = 0; step < numWin; step++) {
      // windowStart: The day index where this evalWin period begins (and rebalancing occurs)
      const windowStart = firstSig + step * currentParams.evalWin;

      // Calculate new portfolio weights using lookBack days of price history ending at windowStart
      const currentWeights = await calculateWindowWeights(allTickerData, dates, windowStart, currentParams, currentStrategy);

      // Calculate transaction costs from trading (buying/selling to match new weights)
      let transactionCost = 0;
      if (prevWeights === null) {
        // First rebalance: cost of initial purchase (all positions are new)
        transactionCost = currentWeights.reduce((sum: number, w: number) => sum + Math.abs(w), 0) * currentParams.tc;
      } else {
        // Subsequent rebalances: cost proportional to weight changes (turnover)
        const turnover = currentWeights.reduce((sum: number, w: number, i: number) => sum + Math.abs(w - (prevWeights![i] || 0)), 0);
        transactionCost = turnover * currentParams.tc;
      }

      // windowEnd: Last day of this evalWin period (before next rebalance)
      const windowEnd = Math.min(windowStart + currentParams.evalWin, dates.length);

      // HOLD PERIOD: Track daily returns while holding the current weights for evalWin days
      for (let day = windowStart; day < windowEnd; day++) {
        if (day === 0) continue;

        // Calculate weighted portfolio return for this day
        let dailyReturn = 0;
        for (let i = 0; i < allTickerData.length; i++) {
          if (day < prices[i].length && prices[i][day - 1] > 0) {
            const stockReturn = (prices[i][day] - prices[i][day - 1]) / prices[i][day - 1];
            dailyReturn += stockReturn * currentWeights[i];
          }
        }

        // Deduct transaction costs on the rebalancing day
        if (day === windowStart) dailyReturn -= transactionCost;

        portfolioReturns.push(dailyReturn);
        portfolioValue *= (1 + dailyReturn);
        nav[dates[day]] = portfolioValue;
      }

      prevWeights = [...currentWeights];
      // Save the last calculated weights - these are displayed in the pie chart
      finalWeights = [...currentWeights];
    }

    const metrics = calculateMetrics(nav, portfolioReturns);

    // Store the FINAL weights from the last rebalancing period.
    // These are what gets displayed in the pie chart and represent the recommended
    // allocation if the user wants to follow this strategy going forward.
    // User action: Allocate portfolio to these weights, hold for evalWin days,
    // then re-run backtest to get updated weights for the next period.
    if (finalWeights) {
      const totalWeight = finalWeights.reduce((sum, weight) => sum + Math.abs(weight), 0);
      if (totalWeight > 0) {
        tickers.forEach((ticker, i) => {
          portfolioWeights[ticker] = Math.abs(finalWeights![i]) / totalWeight;
        });
      } else {
        tickers.forEach((ticker) => {
          portfolioWeights[ticker] = 1.0 / tickers.length;
        });
      }
    }

    return { nav, metrics, portfolioWeights };
  }

  /**
   * Calculate weights for a single evaluation window using AI strategy.
   *
   * This function is called at each rebalancing point. It extracts the lookBack period
   * of historical prices ending at windowStart, then passes this data to the AI strategy
   * to compute optimal portfolio weights for the upcoming evalWin holding period.
   */
  async function calculateWindowWeights(
    allTickerData: Array<{ ticker: string; weight: number; dates: string[]; prices: number[] }>,
    dates: string[],
    windowStart: number,
    currentParams: BacktestParams,
    currentStrategy: GenerationResult | null
  ): Promise<number[]> {
    // Extract lookBack days of price history ending at windowStart for weight calculation
    // Example: if windowStart=35 and lookBack=30, use prices from days 5-34
    const lookbackStart = Math.max(0, windowStart - currentParams.lookBack);
    const lookbackData = allTickerData.map(tickerData => ({
      symbol: tickerData.ticker,
      price: tickerData.prices[windowStart - 1] || tickerData.prices[tickerData.prices.length - 1],
      // Historical prices the AI uses to estimate covariances, trends, momentum, etc.
      lookbackPrices: tickerData.prices.slice(lookbackStart, windowStart),
      lookbackDates: dates.slice(lookbackStart, windowStart)
    }));

    try {
      if (!currentStrategy?.code) throw new Error('No AI strategy code');

      const { executeUserCode } = await import('../../../lib/claude/client');
      const result = await executeUserCode(
        currentStrategy.code,
        'backtest',
        lookbackData,
        undefined,
        undefined,
        {
          backtestDays: currentParams.btHistDays,
          lookbackDays: currentParams.lookBack,
          evaluationWindow: currentParams.evalWin,
          transactionCost: currentParams.tc
        }
      );

      if (result.success && result.weights && Array.isArray(result.weights)) {
        const hasValidWeights = result.weights.every(w => typeof w === 'number' && isFinite(w));
        if (hasValidWeights) {
          return result.weights;
        }
      }
      throw new Error('AI failed to generate valid weights');
    } catch (error) {
      return new Array(allTickerData.length).fill(1.0 / allTickerData.length);
    }
  }

  /**
   * Calculate performance metrics from NAV and returns series.
   */
  function calculateMetrics(
    nav: Record<string, number>,
    portfolioReturns: number[]
  ): Record<string, string | number> {
    const navValues = Object.values(nav);
    const totalReturn = (navValues[navValues.length - 1] - navValues[0]) / navValues[0];
    const avgDailyReturn = portfolioReturns.reduce((sum, ret) => sum + ret, 0) / portfolioReturns.length;
    const dailyVol = Math.sqrt(
      portfolioReturns.reduce((sum, ret) => sum + Math.pow(ret - avgDailyReturn, 2), 0) / (portfolioReturns.length - 1)
    );

    const tradingDays = 252;
    const annualReturn = Math.pow(1 + totalReturn, tradingDays / portfolioReturns.length) - 1;
    const annualVol = dailyVol * Math.sqrt(tradingDays);
    const sharpeRatio = dailyVol > 0 ? (avgDailyReturn / dailyVol * Math.sqrt(tradingDays)) : 0;

    const negativeReturns = portfolioReturns.filter(ret => ret < avgDailyReturn);
    const downsideStd = negativeReturns.length > 0
      ? Math.sqrt(negativeReturns.reduce((sum, ret) => sum + Math.pow(ret - avgDailyReturn, 2), 0) / negativeReturns.length)
      : 0;
    const sortinoRatio = downsideStd > 0 ? (avgDailyReturn / downsideStd * Math.sqrt(tradingDays)) : 0;

    return {
      'Return': `${(totalReturn * 100).toFixed(2)}%`,
      'AnnualReturn': `${(annualReturn * 100).toFixed(2)}%`,
      'DailyVol': `${(dailyVol * 100).toFixed(2)}%`,
      'AnnualVol': `${(annualVol * 100).toFixed(2)}%`,
      'Sharpe': `${sharpeRatio.toFixed(2)}`,
      'Sortino': `${sortinoRatio.toFixed(2)}`
    };
  }

  /**
   * Log backtest results to history endpoint for persistence.
   */
  async function logBacktestResult(
    nav: Record<string, number>,
    portfolioWeights: Record<string, number>,
    metrics: Record<string, string | number>,
    tickers: string[],
    currentParams: BacktestParams,
    currentAlgo: string
  ) {
    try {
      const claudeJobId = `claude-backtest-${Date.now()}`;
      await fetch(`/api/training-logs/${claudeJobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: { status: 'done', nav, weights: portfolioWeights, metrics, algo: currentAlgo.toUpperCase(), tickers },
          originalParams: { algo: currentAlgo.toUpperCase(), tickers, hist_days: currentParams.btHistDays, lookback: currentParams.lookBack, eval_win: currentParams.evalWin, tc: currentParams.tc }
        })
      });
    } catch {
      // Silent fail - don't block backtest completion
    }
  }

  /**
   * Execute traditional algorithm backtest via backend API.
   */
  async function runTraditionalBacktest(
    tickers: string[],
    currentParams: BacktestParams,
    currentAlgo: string
  ) {
    const res = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algo: currentAlgo,
        tickers,
        hist_days: currentParams.btHistDays,
        lookback: currentParams.lookBack,
        eval_win: currentParams.evalWin,
        eta: 0.02,
        tc: currentParams.tc,
      }),
    });
    if (!res.ok) throw new Error(`Backend ${res.status}`);
    const { job_id } = await res.json();
    if (!job_id) throw new Error("No job_id returned");

    let pct = 8;
    const poll = setInterval(async () => {
      try {
        const r = await fetch(`/api/backtest/${job_id}`);
        const data = await r.json();

        if (data.status === "done") {
          clearInterval(poll);
          setResults({ nav: data.nav, weights: data.weights, metrics: data.metrics, usedParams: currentParams });
          setProgress(100);
          setLoading(false);
        } else if (data.status === "error") {
          clearInterval(poll);
          alert(data.detail || "Training failed");
          setLoading(false);
        } else {
          pct = Math.min(pct + 6, 95);
          setProgress(pct);
        }
      } catch {
        clearInterval(poll);
        alert("Lost connection to backend");
        setLoading(false);
      }
    }, 2000);
  }

  return {
    btAlgo,
    setBtAlgo,
    params,
    setParams,
    loading,
    progress,
    results,
    claudeStrategy,
    showPopup,
    openPopup,
    runBacktest,
    handleAlgoChange,
    handleClaudeGenerated,
    handleClaudeError,
    handlePopupClose,
    resetParams,
  };
}
