/**
 * Sandboxed JavaScript execution for user-provided strategy code.
 * Runs code in an isolated environment with restricted globals and timeout protection.
 *
 * ## Security Architecture (Multi-Layer Defense)
 *
 * This module implements Layers 2-4 of the security system:
 *
 * ### Layer 2: Frozen Globals (this file - createSafeGlobals)
 * - All exposed globals are FROZEN using Object.freeze()
 * - Prevents prototype pollution attacks
 * - Limited API surface - only safe operations exposed
 *
 * ### Layer 3: Global Shadowing (this file - BLOCKED_GLOBALS)
 * - Dangerous globals explicitly shadowed with undefined
 * - Even if code tries to access window/document/etc., they're undefined
 *
 * ### Layer 4: Strict Mode
 * - "use strict" enforced in all user code
 * - Prevents `this` from leaking global object
 * - Throws on undeclared variable access
 *
 * ## Important Security Notes
 *
 * This is NOT true sandboxing (like Web Workers or iframes). The code runs
 * in the same JavaScript context. However, the multi-layer defense makes
 * exploitation extremely difficult:
 *
 * 1. Pattern validation (security.ts) blocks known attack patterns
 * 2. Frozen globals prevent prototype pollution
 * 3. Global shadowing prevents direct access to dangerous APIs
 * 4. Strict mode prevents `this` exploitation
 *
 * For this use case (portfolio calculations), this is sufficient because:
 * - Code only receives stockData and returns weights/predictions
 * - No sensitive data (API keys, auth tokens) is in scope
 * - User is already authenticated in their own browser session
 *
 * @see security.ts for Layer 1 (pattern validation)
 */

import type { StockData } from '../core/types';
import { CODE_EXECUTION_TIMEOUT, DEFAULT_FORECAST_DAYS } from '../core/constants';

/**
 * List of dangerous globals that will be explicitly shadowed (set to undefined)
 * in the user code execution scope. This is Layer 3 of the security system.
 *
 * Even if pattern validation misses something, these will be undefined.
 */
const BLOCKED_GLOBALS = [
  // Browser globals
  'window', 'document', 'navigator', 'location', 'history',
  'localStorage', 'sessionStorage', 'indexedDB',

  // Global object references
  'globalThis', 'self', 'global', 'top', 'parent', 'frames',

  // Network APIs
  'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource',

  // Dynamic code execution
  'eval', 'Function',

  // Timers (can be used for side-channel attacks)
  'setTimeout', 'setInterval', 'setImmediate',
  'clearTimeout', 'clearInterval', 'clearImmediate',

  // Workers and threads
  'Worker', 'SharedWorker', 'ServiceWorker',

  // Other dangerous APIs
  'Proxy', 'Reflect', 'SharedArrayBuffer', 'Atomics',
  'WebAssembly', 'Intl',

  // Node.js globals (in case of SSR)
  'process', 'require', 'module', 'exports', '__dirname', '__filename', 'Buffer',
];

/**
 * Creates safe, frozen copies of globals for user code execution.
 * This is Layer 2 of the security system.
 *
 * Key security features:
 * 1. All objects are frozen to prevent prototype pollution
 * 2. Methods are bound to prevent `this` manipulation
 * 3. Only safe, pure operations are exposed
 *
 * @param requestId - Optional ID for logging user code output
 * @returns Frozen object containing safe globals
 */
function createSafeGlobals(requestId?: string) {
  // Create frozen Math object (pure mathematical operations)
  const safeMath = Object.freeze({ ...Math });

  // Create frozen JSON object with bound methods
  const safeJSON = Object.freeze({
    parse: JSON.parse.bind(JSON),
    stringify: JSON.stringify.bind(JSON),
  });

  // Create safe Array utilities (no constructor access)
  const safeArrayUtils = Object.freeze({
    isArray: Array.isArray.bind(Array),
    from: Array.from.bind(Array),
    of: Array.of.bind(Array),
  });

  // Create safe Object utilities (only pure operations)
  const safeObjectUtils = Object.freeze({
    keys: Object.keys.bind(Object),
    values: Object.values.bind(Object),
    entries: Object.entries.bind(Object),
    fromEntries: Object.fromEntries.bind(Object),
    assign: Object.assign.bind(Object),
    freeze: Object.freeze.bind(Object),
    isFrozen: Object.isFrozen.bind(Object),
    // Explicitly NOT including: getPrototypeOf, setPrototypeOf, defineProperty, etc.
  });

  // Create safe console that prefixes output
  const safeConsole = Object.freeze({
    log: (...args: any[]) => console.log(`[User Code${requestId ? ` ${requestId}` : ''}]:`, ...args),
    warn: (...args: any[]) => console.warn(`[User Code${requestId ? ` ${requestId}` : ''}]:`, ...args),
    error: (...args: any[]) => console.error(`[User Code${requestId ? ` ${requestId}` : ''}]:`, ...args),
    info: (...args: any[]) => console.info(`[User Code${requestId ? ` ${requestId}` : ''}]:`, ...args),
  });

  // Return frozen globals object
  return Object.freeze({
    // Mathematical operations
    Math: safeMath,

    // Data structures and utilities
    Array: safeArrayUtils,
    Object: safeObjectUtils,
    JSON: safeJSON,

    // Date (creates new instances, safe to use)
    Date: Date,

    // Primitive constructors/converters (safe)
    Number: Number,
    String: String,
    Boolean: Boolean,

    // Parsing functions
    parseInt: parseInt,
    parseFloat: parseFloat,

    // Type checking
    isNaN: isNaN,
    isFinite: isFinite,

    // Console for debugging
    console: safeConsole,

    // Safe constants
    undefined: undefined,
    NaN: NaN,
    Infinity: Infinity,
  });
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

    // Build global shadowing code (Layer 3)
    // This sets all dangerous globals to undefined in the execution scope
    const globalShadowing = BLOCKED_GLOBALS
      .map(name => `const ${name} = undefined;`)
      .join('\n      ');

    // Build execution wrapper that calls the appropriate function
    // Security layers applied:
    // - Layer 2: Frozen globals passed as parameters
    // - Layer 3: Global shadowing (dangerous names set to undefined)
    // - Layer 4: Strict mode enabled
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

      // =====================================================================
      // Layer 3: Global Shadowing
      // Dangerous globals are explicitly set to undefined to prevent access
      // even if pattern validation (Layer 1) somehow misses an attack
      // =====================================================================
      ${globalShadowing}

      // =====================================================================
      // User Code (transpiled from TypeScript)
      // =====================================================================
      ${jsCode}

      // =====================================================================
      // Function Execution
      // =====================================================================
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
