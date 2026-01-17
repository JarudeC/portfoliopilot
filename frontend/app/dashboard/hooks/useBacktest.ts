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
  lookBack: number;
  evalWin: number;
  tc: number;
}

export interface BacktestResults {
  nav: Record<string, number> | null;
  weights: Record<string, number> | null;
  metrics: Record<string, string | number> | null;
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
    setResults({ nav: null, weights: null, metrics: null });

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
    if (!currentStrategy || !currentStrategy.weights) {
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

    const claudeWeights = currentStrategy.weights;
    let portfolioWeights: Record<string, number> = {};
    tickers.forEach((ticker, i) => {
      portfolioWeights[ticker] = claudeWeights[i] || 0;
    });

    const nav: Record<string, number> = {};
    let metrics: Record<string, string | number> = {};

    try {
      const today = new Date();
      const end = today.toISOString().slice(0, 10);
      const tradingDayMultiplier = 1.43;
      const calendarDaysBack = Math.round(currentParams.btHistDays * tradingDayMultiplier);
      const start = new Date(today.getTime() - calendarDaysBack * 86_400_000).toISOString().slice(0, 10);

      const tickerDataPromises = tickers.map(async (ticker, index) => {
        const res = await fetch(`/api/prices`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, start, end }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (!payload.dates || !payload.prices) throw new Error(`Invalid response for ${ticker}`);

        return {
          ticker,
          weight: claudeWeights[index] || 0,
          dates: payload.dates,
          prices: payload.prices
        };
      });

      const allTickerData = await Promise.all(tickerDataPromises);

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

    setResults({ nav, weights: portfolioWeights, metrics });
    setProgress(100);
    setLoading(false);

    await logBacktestResult(nav, portfolioWeights, metrics, tickers, currentParams, currentAlgo);
  }

  /**
   * Calculate portfolio returns with dynamic AI-driven rebalancing.
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

    const firstSig = currentParams.lookBack;
    const numWin = Math.floor((dates.length - firstSig) / currentParams.evalWin);

    let portfolioValue = 1.0;
    let prevWeights: number[] | null = null;
    let finalWeights: number[] | null = null;
    const portfolioReturns: number[] = [];

    for (let step = 0; step < numWin; step++) {
      const windowStart = firstSig + step * currentParams.evalWin;
      const currentWeights = await calculateWindowWeights(allTickerData, dates, windowStart, currentParams, currentStrategy);

      let transactionCost = 0;
      if (prevWeights === null) {
        transactionCost = currentWeights.reduce((sum: number, w: number) => sum + Math.abs(w), 0) * currentParams.tc;
      } else {
        const turnover = currentWeights.reduce((sum: number, w: number, i: number) => sum + Math.abs(w - (prevWeights![i] || 0)), 0);
        transactionCost = turnover * currentParams.tc;
      }

      const windowEnd = Math.min(windowStart + currentParams.evalWin, dates.length);

      for (let day = windowStart; day < windowEnd; day++) {
        if (day === 0) continue;

        let dailyReturn = 0;
        for (let i = 0; i < allTickerData.length; i++) {
          if (day < prices[i].length && prices[i][day - 1] > 0) {
            const stockReturn = (prices[i][day] - prices[i][day - 1]) / prices[i][day - 1];
            dailyReturn += stockReturn * currentWeights[i];
          }
        }

        if (day === windowStart) dailyReturn -= transactionCost;

        portfolioReturns.push(dailyReturn);
        portfolioValue *= (1 + dailyReturn);
        nav[dates[day]] = portfolioValue;
      }

      prevWeights = [...currentWeights];
      finalWeights = [...currentWeights];
    }

    const metrics = calculateMetrics(nav, portfolioReturns);

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
   */
  async function calculateWindowWeights(
    allTickerData: Array<{ ticker: string; weight: number; dates: string[]; prices: number[] }>,
    dates: string[],
    windowStart: number,
    currentParams: BacktestParams,
    currentStrategy: GenerationResult | null
  ): Promise<number[]> {
    const lookbackStart = Math.max(0, windowStart - currentParams.lookBack);
    const lookbackData = allTickerData.map(tickerData => ({
      symbol: tickerData.ticker,
      price: tickerData.prices[windowStart - 1] || tickerData.prices[tickerData.prices.length - 1],
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
      await fetch(`/api/train/${claudeJobId}`, {
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
    const res = await fetch("/api/train", {
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
        const r = await fetch(`/api/train/${job_id}`);
        const data = await r.json();

        if (data.status === "done") {
          clearInterval(poll);
          setResults({ nav: data.nav, weights: data.weights, metrics: data.metrics });
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
    runBacktest,
    handleAlgoChange,
    handleClaudeGenerated,
    handleClaudeError,
    handlePopupClose,
    resetParams,
  };
}
