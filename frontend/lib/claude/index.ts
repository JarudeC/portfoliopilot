/**
 * Claude AI integration barrel export.
 *
 * Module structure:
 * - core/: Shared types, constants, and error classes
 * - client/: Client-side API wrapper with rate limiting and request management
 * - server/: Server-side code generation and execution (used by API routes)
 * - execution/: Secure code execution and security validation
 */

// ============================================================================
// Core Module (Types, Constants, Errors)
// ============================================================================

export type {
  StockData,
  GenerationResult,
  ValidationResult,
  ExecutionResult,
  DashboardParams,
} from './core/types';

export type { ProgressState } from './core/constants';

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
  // Defaults
  DEFAULT_FORECAST_DAYS,
  // Trend multipliers
  TREND_UPWARD,
  TREND_VOLATILE_RANGE,
  TREND_SIDEWAYS_RANGE,
  TREND_DOWNWARD,
  TREND_RANDOM_RANGE,
  // Progress states
  PROGRESS_STATES,
} from './core/constants';

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
  ErrorUtils,
} from './core/errors';

// ============================================================================
// Client Module (Browser-side API)
// ============================================================================

export {
  generateCodeOnly,
  executeUserCode,
} from './client/client';

export type {
  GenerateRequest,
  GenerateResponse,
} from './client/client';

export { requestTracker } from './client/request-tracker';

// ============================================================================
// Server Module (AI Generation)
// ============================================================================

export {
  generatePortfolioWeights,
  generateCodeOnly as generateCodeOnlyServer,
  executeUserCode as executeUserCodeServer,
} from './server/generator';

export {
  createRigidPrompt,
  extractTypeScriptCode,
  stripTypeScriptTypes,
  validateAndFixCode,
  BACKTEST_FALLBACK_FUNCTION,
  FORECAST_FALLBACK_FUNCTION,
} from './server/code-processing';

export {
  executeWithTimeout,
  generateFallbackResult,
} from './server/fallback';

// ============================================================================
// Execution Module (Security & Sandbox)
// ============================================================================

export {
  executeJavaScriptLocally,
  transpileTypeScriptToJavaScript,
} from './execution/code-sandbox';

export {
  validateSecurity,
  createSecurityConfig,
} from './execution/security';

export type {
  SecurityConfig,
  ValidationResult as SecurityValidationResult,
  PromptValidationResult,
  CodeValidationResult,
} from './execution/security';
