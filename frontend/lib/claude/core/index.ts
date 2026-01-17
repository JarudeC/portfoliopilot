/**
 * Core module - shared foundations for Claude AI integration.
 * Contains types, constants, and error handling.
 */

// Types
export type {
  StockData,
  GenerationResult,
  ValidationResult,
  ExecutionResult,
  DashboardParams,
} from './types';

// Constants
export {
  // Timeouts
  CODE_EXECUTION_TIMEOUT,
  API_REQUEST_TIMEOUT,
  RATE_LIMIT_WINDOW,
  LOADING_CLEANUP_DELAY,
  RETRY_DELAY_BASE,
  // Limits
  MAX_REQUESTS_PER_WINDOW,
  DEFAULT_MAX_RETRIES,
  MAX_DESCRIPTION_LENGTH,
  MAX_STOCK_COUNT,
  // Defaults
  DEFAULT_FORECAST_DAYS,
  DEFAULT_STOCK_DATA,
  // Trend multipliers
  TREND_UPWARD,
  TREND_VOLATILE_RANGE,
  TREND_SIDEWAYS_RANGE,
  TREND_DOWNWARD,
  TREND_RANDOM_RANGE,
  // Progress
  PROGRESS_STATES,
} from './constants';

export type { ProgressState } from './constants';

// Errors
export {
  ErrorSeverity,
  ErrorCategory,
  ClaudeError,
  ClaudeApiError,
  NetworkError,
  ValidationError,
  ExecutionError,
  TimeoutError,
  RateLimitError,
  ParseError,
  SecurityError,
  ErrorFactory,
  ErrorAggregator,
  ErrorUtils,
} from './errors';
