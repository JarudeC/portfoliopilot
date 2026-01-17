/**
 * Client module - browser-side API client for Claude AI.
 */

// Main client functions
export {
  generateStrategy,
  generateCodeOnly,
  executeUserCode,
} from './client';

export type {
  GenerateRequest,
  GenerateResponse,
  ApiErrorResponse,
  LoadingState,
  StockData,
  GenerationResult,
  SecurityConfig,
} from './client';

// Request tracking
export {
  RequestTracker,
  requestTracker,
} from './request-tracker';

// Loading state management
export {
  LoadingStateManager,
  loadingStateManager,
  loadingHelpers,
} from './loading-state';
