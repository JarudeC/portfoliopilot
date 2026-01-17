/**
 * Dashboard hooks barrel export.
 */
export { useBacktest, BACKTEST_ALGOS, LOOKBACKS, EVALWINS, BTHIST_DAYS } from './useBacktest';
export type { BacktestParams, BacktestResults, UseBacktestReturn } from './useBacktest';

export { useForecast, FORECAST_ALGOS, HIST_DAYS, FORECAST_DAYS } from './useForecast';
export type { ForecastParams, UseForecastReturn } from './useForecast';
