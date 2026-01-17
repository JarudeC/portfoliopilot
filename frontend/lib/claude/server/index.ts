/**
 * Server module - AI strategy generation and code processing.
 * Used server-side for Claude API interactions.
 */

// Generator (main entry points)
export {
  generatePortfolioWeights,
  generateCodeOnly,
  executeUserCode,
} from './generator';

// Code processing
export {
  createRigidPrompt,
  extractTypeScriptCode,
  stripTypeScriptTypes,
  validateAndFixCode,
  BACKTEST_FALLBACK_FUNCTION,
  FORECAST_FALLBACK_FUNCTION,
} from './code-processing';

// Fallback strategies
export {
  executeWithTimeout,
  generateFallbackResult,
} from './fallback';
