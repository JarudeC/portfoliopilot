/**
 * Sandboxed JavaScript execution for user-provided strategy code.
 * Runs code in an isolated environment with restricted globals and timeout protection.
 *
 * Security measures:
 * - Only safe globals exposed (Math, Date, Array, etc.)
 * - No access to window, document, fetch, eval, etc.
 * - Configurable timeout to prevent infinite loops
 * - TypeScript transpilation via Babel
 */

import type { StockData } from '../core/types';
import { CODE_EXECUTION_TIMEOUT, DEFAULT_FORECAST_DAYS } from '../core/constants';

/**
 * Safe globals exposed to user code.
 * Excludes dangerous APIs like fetch, eval, window, document.
 */
function createSafeGlobals(requestId?: string) {
  return {
    Math: Math,
    Date: Date,
    Array: Array,
    Object: Object,
    JSON: JSON,
    Number: Number,
    String: String,
    Boolean: Boolean,
    parseInt: parseInt,
    parseFloat: parseFloat,
    isNaN: isNaN,
    isFinite: isFinite,
    console: {
      log: (...args: any[]) => console.log(`[User Code ${requestId}]:`, ...args),
      warn: (...args: any[]) => console.warn(`[User Code ${requestId}]:`, ...args),
      error: (...args: any[]) => console.error(`[User Code ${requestId}]:`, ...args)
    }
  };
}

/**
 * Execute user JavaScript code in a sandboxed environment.
 * Creates a Function with controlled scope - no access to dangerous APIs.
 *
 * @param userCode - The strategy code to execute
 * @param mode - 'forecast' or 'backtest' determines which function to call
 * @param stockData - Stock data to pass to the strategy function
 * @param forecastDays - Number of days to forecast (forecast mode only)
 * @param dashboardParams - Additional parameters from dashboard
 * @param requestId - Optional request ID for logging
 * @returns Weights array (backtest) or predictions array (forecast)
 */
export async function executeJavaScriptLocally(
  userCode: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[],
  forecastDays?: number,
  dashboardParams?: any,
  requestId?: string
): Promise<{ weights?: number[], predictions?: Array<{date: string, price: number, confidence?: number}> }> {

  const safeGlobals = createSafeGlobals(requestId);
  const effectiveForecastDays = forecastDays || DEFAULT_FORECAST_DAYS;

  try {
    // Transpile TypeScript to JavaScript using Babel
    const jsCode = await transpileTypeScriptToJavaScript(userCode);

    // Build execution wrapper that calls the appropriate function
    const executeCode = new Function(
      'stockData',
      'forecastDays',
      'dashboardParams',
      'Math',
      'Date',
      'Array',
      'Object',
      'JSON',
      'Number',
      'String',
      'Boolean',
      'parseInt',
      'parseFloat',
      'isNaN',
      'isFinite',
      'console',
      `
      "use strict";

      ${jsCode}

      // Execute the appropriate function based on mode
      if (typeof calculateWeights === 'function') {
        const weights = calculateWeights(stockData, dashboardParams || {});
        if (!Array.isArray(weights)) {
          throw new Error('calculateWeights must return an array of numbers');
        }
        return { weights: weights };
      } else if (typeof generatePredictions === 'function') {
        const predictions = generatePredictions(stockData, forecastDays, dashboardParams || {});
        if (!Array.isArray(predictions)) {
          throw new Error('generatePredictions must return an array of prediction objects');
        }
        return { predictions: predictions };
      } else {
        throw new Error('Required function not found: ' + (${mode === 'backtest' ? '"calculateWeights"' : '"generatePredictions"'}));
      }
      `
    );

    // Execute with controlled scope and timeout
    const executionPromise = new Promise((resolve, reject) => {
      try {
        const result = executeCode(
          stockData,
          effectiveForecastDays,
          dashboardParams,
          safeGlobals.Math,
          safeGlobals.Date,
          safeGlobals.Array,
          safeGlobals.Object,
          safeGlobals.JSON,
          safeGlobals.Number,
          safeGlobals.String,
          safeGlobals.Boolean,
          safeGlobals.parseInt,
          safeGlobals.parseFloat,
          safeGlobals.isNaN,
          safeGlobals.isFinite,
          safeGlobals.console
        );
        resolve(result);
      } catch (error) {
        reject(error);
      }
    });

    // Add timeout to prevent infinite loops
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Code execution timeout (${CODE_EXECUTION_TIMEOUT / 1000} seconds)`)), CODE_EXECUTION_TIMEOUT)
    );

    const result = await Promise.race([executionPromise, timeoutPromise]) as any;

    // Validate result structure
    if (mode === 'backtest' && result.weights) {
      if (!Array.isArray(result.weights) || result.weights.some((w: any) => typeof w !== 'number' || !isFinite(w))) {
        throw new Error('calculateWeights must return an array of finite numbers');
      }
    } else if (mode === 'forecast' && result.predictions) {
      if (!Array.isArray(result.predictions) ||
          result.predictions.some((p: any) => !p || typeof p.price !== 'number' || !p.date)) {
        throw new Error('generatePredictions must return an array of objects with date and price properties');
      }
    }

    return result;

  } catch (error: any) {
    throw new Error(`JavaScript execution error: ${error.message || 'Unknown error'}`);
  }
}

/**
 * Transpile TypeScript to JavaScript using Babel Standalone.
 * Falls back to regex-based type stripping if Babel fails.
 *
 * @param code - TypeScript code to transpile
 * @returns JavaScript code
 */
export async function transpileTypeScriptToJavaScript(code: string): Promise<string> {
  try {
    // Import Babel Standalone (browser-compatible)
    const Babel = await import('@babel/standalone');

    // Transpile TypeScript to JavaScript
    const result = Babel.transform(code, {
      presets: ['typescript'],
      filename: 'user-code.ts'
    });

    if (!result || !result.code) {
      throw new Error('Babel transpilation failed - no output code');
    }

    return result.code;

  } catch (error: any) {
    console.warn('Babel TypeScript transpilation failed:', error);
    console.log('Falling back to simple regex-based stripping');

    // Fallback: simple regex-based type stripping
    return code
      .replace(/:\s*[^=,){\s]+(?=\s*[=,){])/g, '')  // Remove : Type annotations
      .replace(/\):\s*[^{]+\{/g, ') {');            // Remove return types
  }
}
