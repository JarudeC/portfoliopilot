/**
 * Centralized constants for the Claude AI integration.
 * All magic numbers, timeouts, and default values are defined here.
 */

// ============================================================================
// Timeouts (milliseconds)
// ============================================================================

/** Timeout for executing user code in sandbox (5 seconds) */
export const CODE_EXECUTION_TIMEOUT = 5000;

/** Timeout for API requests to Claude (30 seconds) */
export const API_REQUEST_TIMEOUT = 30000;

/** Rate limit window duration (1 minute) */
export const RATE_LIMIT_WINDOW = 60000;

/** Delay before cleaning up loading state (1 second) */
export const LOADING_CLEANUP_DELAY = 1000;

/** Base delay for retry logic (1 second) */
export const RETRY_DELAY_BASE = 1000;

// ============================================================================
// Limits
// ============================================================================

/** Maximum requests allowed per rate limit window */
export const MAX_REQUESTS_PER_WINDOW = 8;

/** Maximum retry attempts for failed requests */
export const DEFAULT_MAX_RETRIES = 3;

/** Maximum length for strategy description */
export const MAX_DESCRIPTION_LENGTH = 2000;

/** Maximum number of stocks that can be processed */
export const MAX_STOCK_COUNT = 100;

// ============================================================================
// Defaults
// ============================================================================

/** Default number of days for forecast predictions */
export const DEFAULT_FORECAST_DAYS = 30;

/** Default stock data for testing when none provided */
export const DEFAULT_STOCK_DATA = [
  { symbol: 'SPY', price: 400 },
  { symbol: 'QQQ', price: 350 },
  { symbol: 'VTI', price: 200 },
] as const;

// ============================================================================
// Fallback Trend Multipliers
// Used in fallback.ts for generating fallback predictions
// ============================================================================

/** Upward trend: +0.1% daily growth */
export const TREND_UPWARD = 1.001;

/** Volatile trend: ±2% daily swing */
export const TREND_VOLATILE_RANGE = 0.04;

/** Sideways trend: ±0.5% daily change */
export const TREND_SIDEWAYS_RANGE = 0.01;

/** Downward trend: -0.1% daily decline */
export const TREND_DOWNWARD = 0.999;

/** Random walk: ±1% daily change */
export const TREND_RANDOM_RANGE = 0.02;

// ============================================================================
// Progress States
// ============================================================================

export type ProgressState = 'validating' | 'generating' | 'processing';

export const PROGRESS_STATES = {
  VALIDATING: 'validating' as const,
  GENERATING: 'generating' as const,
  PROCESSING: 'processing' as const,
};
