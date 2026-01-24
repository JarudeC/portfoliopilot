/**
 * Client module - browser-side API client for Claude AI.
 */

// Main client functions
export {
  generateCodeOnly,
  executeUserCode,
} from './client';

export type {
  GenerateRequest,
  GenerateResponse,
  ApiErrorResponse,
  StockData,
  GenerationResult,
  SecurityConfig,
} from './client';

// Request tracking
export { requestTracker } from './request-tracker';
