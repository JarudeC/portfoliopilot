/**
 * Core types for Claude AI strategy generation.
 * Shared interfaces used across client, generator, and execution modules.
 */

/**
 * Stock data passed to strategy functions.
 * For backtest: includes lookbackPrices/lookbackDates for historical analysis.
 * For forecast: contains current price data only.
 */
export interface StockData {
  symbol: string;
  price: number;
  marketCap?: number;
  volume?: number;
  change?: number;
  changePercent?: number;
  /** Historical prices for lookback period (backtest only) */
  lookbackPrices?: number[];
  /** Dates corresponding to lookbackPrices (backtest only) */
  lookbackDates?: string[];
  [key: string]: any;
}

/**
 * Result from strategy generation or execution.
 * Contains either weights (backtest) or predictions (forecast).
 */
export interface GenerationResult {
  success: boolean;
  type: 'forecast' | 'backtest';
  /** Portfolio weights for each stock (backtest mode) */
  weights?: number[];
  /** Price predictions with dates (forecast mode) */
  predictions?: Array<{
    date: string;
    price: number;
    confidence?: number;
  }>;
  /** Generated or executed code */
  code?: string;
  error?: string;
  /** True if AI generation failed and fallback was used */
  fallbackUsed: boolean;
  /** True if strategy was loaded from saved strategies */
  loadedFromSaved?: boolean;
  executionTime: number;
  securityValidation?: {
    promptValid: boolean;
    codeValid?: boolean;
    blockedReason?: string;
    riskLevel: 'none' | 'low' | 'medium' | 'high';
  };
}

/**
 * Result from code validation.
 */
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  fixedCode?: string;
}

/**
 * Result from sandboxed code execution.
 */
export interface ExecutionResult {
  success: boolean;
  result?: number[] | any[];
  error?: string;
  timeout: boolean;
}

/**
 * Dashboard parameters passed to strategy functions.
 */
export interface DashboardParams {
  backtestDays?: number;
  lookbackDays?: number;
  evaluationWindow?: number;
  transactionCost?: number;
  historyDays?: number;
  forecastDays?: number;
}
