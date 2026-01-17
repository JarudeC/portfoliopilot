/**
 * Code generation and processing for Claude AI strategies.
 * Handles prompt creation, code extraction from Claude responses, and validation.
 */

import type { ValidationResult } from '../core/types';

// ============================================================================
// PROMPT TEMPLATES
// ============================================================================

/**
 * Fallback function when AI generation fails (backtest mode).
 * Returns equal weights for all stocks.
 */
export const BACKTEST_FALLBACK_FUNCTION = `
function calculateWeights(stockData: any[], dashboardParams: any = {}): number[] {
  const count = stockData.length || 1;
  const weight = 1.0 / count;
  return new Array(count).fill(weight);
}
`;

/**
 * Fallback function when AI generation fails (forecast mode).
 * Returns simple random walk predictions.
 */
export const FORECAST_FALLBACK_FUNCTION = `
function generatePredictions(stockData: any[], days: number = 30, dashboardParams: any = {}): Array<{date: string, price: number, confidence?: number}> {
  const predictions = [];
  const startDate = new Date();

  for (let i = 1; i <= days; i++) {
    const futureDate = new Date(startDate);
    futureDate.setDate(startDate.getDate() + i);

    // Simple trend prediction - return percentage multiplier
    const trend = 1 + (Math.random() - 0.5) * 0.02; // ±1% daily change
    const cumulativeMultiplier = Math.pow(trend, i);

    predictions.push({
      date: futureDate.toISOString().split('T')[0],
      price: cumulativeMultiplier,
      confidence: Math.max(0.3, 0.9 - (i * 0.02))
    });
  }

  return predictions;
}
`;

/**
 * Prompt template for backtest weight calculation.
 * Instructs Claude to generate a calculateWeights function.
 */
const BACKTEST_PROMPT_TEMPLATE = `
You are a TypeScript code generator. You MUST respond with ONLY a valid TypeScript function following these EXACT requirements:

1. Function name MUST be exactly "calculateWeights"
2. Function MUST accept parameters: stockData: any[], dashboardParams: any = {}
3. Function MUST return: number[]
4. Function MUST be a complete, standalone TypeScript function
5. Do NOT include any explanatory text, markdown, or code blocks
6. Do NOT include imports, exports, or interface definitions
7. Return ONLY the function code

IMPORTANT - stockData structure for DYNAMIC REBALANCING:
Each stock object contains: {symbol: string, price: number, lookbackPrices: number[], lookbackDates: string[]}
- symbol: the stock ticker (e.g., "AAPL")
- price: current price at rebalancing time
- lookbackPrices: array of historical prices for the lookback period (ALWAYS available, use this for calculations)
- lookbackDates: corresponding dates for lookbackPrices (ALWAYS available)
- Use lookbackPrices to calculate: returns, volatility, momentum, moving averages, trends
- Example: daily returns = (lookbackPrices[i] - lookbackPrices[i-1]) / lookbackPrices[i-1]

The dashboardParams object contains:
- backtestDays: total historical days for backtesting
- lookbackDays: lookback period for analysis (data window size)
- evaluationWindow: rebalancing frequency in days
- transactionCost: transaction cost rate applied when weights change

Your function will be called at EACH rebalancing period with fresh market data.
Analyze the lookback data to make dynamic allocation decisions based on recent market conditions.

The function should implement this portfolio strategy: {userDescription}

CRITICAL REQUIREMENTS:
1. ENSURE PERFECT SYNTAX - Code must be valid JavaScript/TypeScript with no syntax errors
2. DOUBLE-CHECK all variable names, parentheses, brackets, and semicolons
3. DO NOT normalize weights to sum to 1 - return raw scores/weights
4. DO NOT use equal weights (1/n) as fallback under ANY circumstances EXCEPT if the user explicitly asks for it
5. DO NOT include any code that divides by total weight sum
6. Return weights that reflect your actual analysis - if one stock scores 2x higher, it should get 2x the weight
7. Return only NON-NEGATIVE weights (>= 0). Zero weights are acceptable for poor-performing stocks
8. DO NOT return negative weights - use 0 instead for stocks you want to avoid
9. The system handles all scaling - your job is differentiation based on analysis
10. VERIFY: Every variable used must be properly declared and defined

Generate the calculateWeights function now:
`;

/**
 * Prompt template for forecast prediction generation.
 * Instructs Claude to generate a generatePredictions function.
 */
const FORECAST_PROMPT_TEMPLATE = `
You are a TypeScript code generator. You MUST respond with ONLY a valid TypeScript function following these EXACT requirements:

1. Function name MUST be exactly "generatePredictions"
2. Function MUST accept parameters: stockData: any[], days: number = 30, dashboardParams: any = {}
3. Function MUST return: Array<{date: string, price: number, confidence?: number}>
4. Function MUST be a complete, standalone TypeScript function
5. Do NOT include any explanatory text, markdown, or code blocks
6. Do NOT include imports, exports, or interface definitions
7. Return ONLY the function code

IMPORTANT - stockData structure:
Each stock is an object with: {symbol: string, price: number}
- symbol: the stock ticker (e.g., "AAPL")
- price: current price of the stock
Note: You only have current price data for forecasting. Use mathematical models to project future prices.

The dashboardParams object contains:
- historyDays: number of historical days available for analysis
- forecastDays: number of days to forecast (same as 'days' parameter)

The function should implement this forecasting strategy: {userDescription}

Each prediction object should have:
- date: ISO date string (YYYY-MM-DD format)
- price: predicted price as PERCENTAGE CHANGE from starting price (e.g., 1.02 means +2%, 0.98 means -2%)
- confidence: optional confidence score between 0 and 1

IMPORTANT: Return percentage multipliers, NOT absolute prices.

Generate the generatePredictions function now:
`;

// ============================================================================
// PROMPT CREATION
// ============================================================================

/**
 * Create a rigid prompt for Claude AI based on user description.
 *
 * @param userDescription - User's strategy description
 * @param mode - 'forecast' or 'backtest'
 * @returns Formatted prompt string
 */
export function createRigidPrompt(userDescription: string, mode: 'forecast' | 'backtest'): string {
  const template = mode === 'forecast' ? FORECAST_PROMPT_TEMPLATE : BACKTEST_PROMPT_TEMPLATE;
  return template
    .replace(/{userDescription}/g, userDescription)
    .trim();
}

// ============================================================================
// CODE EXTRACTION
// ============================================================================

/**
 * Extract TypeScript function code from Claude's response.
 * Handles markdown code blocks and finds the target function.
 *
 * @param response - Raw response from Claude API
 * @param mode - 'forecast' or 'backtest' determines which function to find
 * @returns Extracted function code or fallback
 */
export function extractTypeScriptCode(response: string, mode: 'forecast' | 'backtest'): string {
  let cleaned = response.trim();

  // Remove markdown code blocks
  cleaned = cleaned.replace(/```typescript\n?/g, '');
  cleaned = cleaned.replace(/```ts\n?/g, '');
  cleaned = cleaned.replace(/```javascript\n?/g, '');
  cleaned = cleaned.replace(/```js\n?/g, '');
  cleaned = cleaned.replace(/```\n?/g, '');

  const lines = cleaned.split('\n');
  let functionStart = -1;
  let functionEnd = -1;
  let braceCount = 0;

  // Look for the appropriate function based on mode
  const functionName = mode === 'forecast' ? 'generatePredictions' : 'calculateWeights';

  // Find function start
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.includes(`function ${functionName}`) || line.includes(`${functionName} =`) || line.includes(`const ${functionName}`)) {
      functionStart = i;
      break;
    }
  }

  if (functionStart === -1) {
    return mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;
  }

  // Find function end by matching braces
  for (let i = functionStart; i < lines.length; i++) {
    const line = lines[i];
    for (const char of line) {
      if (char === '{') braceCount++;
      if (char === '}') braceCount--;
    }
    if (braceCount === 0 && line.includes('}')) {
      functionEnd = i;
      break;
    }
  }

  if (functionEnd === -1) {
    return mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;
  }

  return lines.slice(functionStart, functionEnd + 1).join('\n');
}

// ============================================================================
// TYPE STRIPPING
// ============================================================================

/**
 * Strip TypeScript type annotations for JavaScript execution.
 * Used as fallback when Babel transpilation fails.
 *
 * @param code - TypeScript code
 * @returns JavaScript code with types removed
 */
export function stripTypeScriptTypes(code: string): string {
  let jsCode = code;

  // Remove return type annotations
  jsCode = jsCode.replace(/\):\s*Array<[^>]*>\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*number\[\]\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*string\[\]\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*any\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*number\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*string\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*boolean\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*void\s*\{/g, ') {');

  // Remove parameter type annotations
  jsCode = jsCode.replace(/function\s+(\w+)\s*\([^)]*\)/g, (match) => {
    let cleanedMatch = match;

    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*any\[\]/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*number\[\]/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*string\[\]/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*any/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*number/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*string/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*boolean/g, '$1');
    cleanedMatch = cleanedMatch.replace(/(\w+)\s*:\s*Array<[^>]*>/g, '$1');

    return cleanedMatch;
  });

  return jsCode;
}

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * Validate and fix generated code structure.
 * Ensures function signature and parameters are correct.
 *
 * @param code - Code to validate
 * @param mode - 'forecast' or 'backtest'
 * @returns Validation result with potential fixes
 */
export function validateAndFixCode(code: string, mode: 'forecast' | 'backtest'): ValidationResult {
  const errors: string[] = [];
  let fixedCode = code;

  const functionName = mode === 'forecast' ? 'generatePredictions' : 'calculateWeights';
  const fallbackFunction = mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;

  // Check function name exists
  if (!code.includes(functionName)) {
    errors.push(`Function name must be ${functionName}`);
    return { isValid: false, errors };
  }

  // Check stockData parameter
  if (!code.includes('stockData: any[]') && !code.includes('stockData:any[]')) {
    if (code.includes('stockData')) {
      fixedCode = fixedCode.replace(/stockData:\s*\w+(\[\])?/g, 'stockData: any[]');
    } else {
      errors.push('Missing stockData parameter');
    }
  }

  // Check dashboardParams parameter
  if (!code.includes('dashboardParams')) {
    if (code.includes('stockData: any[]')) {
      fixedCode = fixedCode.replace(/stockData: any\[\]/, 'stockData: any[], dashboardParams: any = {}');
    }
  }

  // Mode-specific validation
  if (mode === 'backtest') {
    // Validate return type for backtest
    if (!code.includes(': number[]')) {
      if (code.includes('): number') && !code.includes('): number[]')) {
        fixedCode = fixedCode.replace(/\):\s*number/g, '): number[]');
      } else if (!code.includes('): number')) {
        fixedCode = fixedCode.replace(/\)\s*\{/, '): number[] {');
      }
    }

    const functionRegex = /function\s+calculateWeights\s*\([^)]*stockData[^)]*\)\s*:\s*number\[\]\s*{[\s\S]*}/;
    const arrowRegex = /const\s+calculateWeights\s*=\s*\([^)]*stockData[^)]*\)\s*:\s*number\[\]\s*=>\s*{[\s\S]*}/;

    if (!functionRegex.test(fixedCode) && !arrowRegex.test(fixedCode)) {
      errors.push('Invalid backtest function structure');
      return { isValid: false, errors, fixedCode: fallbackFunction };
    }
  } else {
    // Validate return type for forecast
    if (!code.includes(': Array<')) {
      if (code.includes('): ') && !code.includes(': Array<')) {
        fixedCode = fixedCode.replace(/\):\s*\w+[\[\]]*/, '): Array<{date: string, price: number, confidence?: number}>');
      } else if (!code.includes('): ')) {
        fixedCode = fixedCode.replace(/\)\s*\{/, '): Array<{date: string, price: number, confidence?: number}> {');
      }
    }

    const functionRegex = /function\s+generatePredictions\s*\([^)]*stockData[^)]*\)\s*:\s*Array<[^>]*>\s*{[\s\S]*}/;
    const arrowRegex = /const\s+generatePredictions\s*=\s*\([^)]*stockData[^)]*\)\s*:\s*Array<[^>]*>\s*=>\s*{[\s\S]*}/;

    if (!functionRegex.test(fixedCode) && !arrowRegex.test(fixedCode)) {
      errors.push('Invalid forecast function structure');
      return { isValid: false, errors, fixedCode: fallbackFunction };
    }
  }

  return { isValid: errors.length === 0, errors, fixedCode };
}
