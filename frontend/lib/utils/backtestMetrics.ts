/**
 * Backtest performance metrics calculation.
 * Used by Custom AI backtesting (frontend-side calculation).
 * Classical algorithms get metrics from the backend.
 */

export interface BacktestMetrics extends Record<string, string | number> {
  Return: string;
  AnnualReturn: string;
  DailyVol: string;
  AnnualVol: string;
  Sharpe: string;
  Sortino: string;
}

/**
 * Calculate backtest performance metrics from NAV and returns.
 *
 * @param nav - Record of date -> NAV value
 * @param portfolioReturns - Array of daily returns
 * @returns Formatted metrics object
 */
export function calculateBacktestMetrics(
  nav: Record<string, number>,
  portfolioReturns: number[]
): BacktestMetrics {
  const navValues = Object.values(nav);
  const totalReturn = (navValues[navValues.length - 1] - navValues[0]) / navValues[0];
  const avgDailyReturn = portfolioReturns.reduce((sum, ret) => sum + ret, 0) / portfolioReturns.length;
  const dailyVol = Math.sqrt(
    portfolioReturns.reduce((sum, ret) => sum + Math.pow(ret - avgDailyReturn, 2), 0) / (portfolioReturns.length - 1)
  );

  const tradingDays = 252; //Represents number of trading days in a year
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
