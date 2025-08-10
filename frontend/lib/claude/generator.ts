import { validateSecurity, SecurityConfig } from './security';

export interface StockData {
  symbol: string;
  price: number;
  marketCap?: number;
  volume?: number;
  change?: number;
  changePercent?: number;
  [key: string]: any;
}

export interface GenerationResult {
  success: boolean;
  type: 'forecast' | 'backtest';
  // For backtest mode
  weights?: number[];
  // For forecast mode  
  predictions?: Array<{
    date: string;
    price: number;
    confidence?: number;
  }>;
  code?: string;
  error?: string;
  fallbackUsed: boolean;
  executionTime: number;
  securityValidation?: {
    promptValid: boolean;
    codeValid?: boolean;
    blockedReason?: string;
    riskLevel: 'none' | 'low' | 'medium' | 'high';
  };
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  fixedCode?: string;
}

export interface ExecutionResult {
  success: boolean;
  result?: number[];
  error?: string;
  timeout: boolean;
}

const BACKTEST_FALLBACK_FUNCTION = `
function calculateWeights(stockData: any[], dashboardParams: any = {}): number[] {
  const count = stockData.length || 1;
  const weight = 1.0 / count;
  return new Array(count).fill(weight);
}
`;

const FORECAST_FALLBACK_FUNCTION = `
function generatePredictions(stockData: any[], days: number = 30, dashboardParams: any = {}): Array<{date: string, price: number, confidence?: number}> {
  const predictions = [];
  const startDate = new Date();
  
  for (let i = 1; i <= days; i++) {
    const futureDate = new Date(startDate);
    futureDate.setDate(startDate.getDate() + i);
    
    // Simple trend prediction with some randomness - return percentage multiplier
    const trend = 1 + (Math.random() - 0.5) * 0.02; // ±1% daily change
    const cumulativeMultiplier = Math.pow(trend, i);
    
    predictions.push({
      date: futureDate.toISOString().split('T')[0],
      price: cumulativeMultiplier, // This is a multiplier, not absolute price
      confidence: Math.max(0.3, 0.9 - (i * 0.02))
    });
  }
  
  return predictions;
}
`;

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
Each stock object contains: {symbol: string, price: number, lookbackPrices?: number[], lookbackDates?: string[], marketCap?: number, volume?: number}
- price: current price at rebalancing time
- lookbackPrices: array of historical prices for the lookback period
- lookbackDates: corresponding dates for lookbackPrices
- You can analyze trends, volatility, momentum using lookbackPrices/lookbackDates

The dashboardParams object contains:
- backtestDays: total historical days for backtesting
- lookbackDays: lookback period for analysis (data window size)
- evaluationWindow: rebalancing frequency in days  
- transactionCost: transaction cost rate applied when weights change

Your function will be called at EACH rebalancing period with fresh market data.
Analyze the lookback data to make dynamic allocation decisions based on recent market conditions.

The function should implement this portfolio strategy: {userDescription}

You can implement sophisticated strategies using:
- Momentum analysis (price trends from lookbackPrices)
- Mean reversion (price vs historical average)  
- Volatility-based allocation (risk parity)
- Technical indicators (moving averages, RSI, etc.)
- Market cap and fundamental weighting
- Multi-factor approaches

Example format:
function calculateWeights(stockData: any[], dashboardParams: any = {}): number[] {
  // Implementation based on: {userDescription}
  // Analyze stockData[i].lookbackPrices for trends, volatility
  // Use stockData[i].price for current valuation
  // Consider dashboardParams.transactionCost for turnover management
  
  const weights = stockData.map(stock => {
    // Example: momentum strategy
    if (stock.lookbackPrices && stock.lookbackPrices.length > 10) {
      const recentAvg = stock.lookbackPrices.slice(-10).reduce((a,b) => a+b) / 10;
      const momentum = stock.price / recentAvg - 1;
      return Math.max(0, momentum); // Positive momentum weighting
    }
    return 1.0 / stockData.length; // Fallback to equal weight
  });
  
  // Normalize weights
  const sum = weights.reduce((a,b) => a+b, 0);
  return sum > 0 ? weights.map(w => w/sum) : stockData.map(() => 1.0/stockData.length);
}

Generate the calculateWeights function now:
`;

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
Each stock is an object with CURRENT data only: {symbol: string, price: number, marketCap?: number, volume?: number}
You do NOT have historical price data or dates. Only current price, marketCap, and volume.

The dashboardParams object contains:
- historyDays: number of historical days available for analysis
- forecastDays: number of days to forecast (same as 'days' parameter)

The function should implement this forecasting strategy: {userDescription}

Since you only have current data, create predictions using:
- Current price as starting point
- Market cap or volume as trend indicators
- Mathematical models (random walk, trends, cycles)
- Price relationships between stocks

Each prediction object should have:
- date: ISO date string (YYYY-MM-DD format)
- price: predicted price as PERCENTAGE CHANGE from starting price (e.g., 1.02 means +2%, 0.98 means -2%)
- confidence: optional confidence score between 0 and 1

IMPORTANT: Return percentage multipliers, NOT absolute prices. This allows the same forecast to work for any stock.

Example format:
function generatePredictions(stockData: any[], days: number = 30, dashboardParams: any = {}): Array<{date: string, price: number, confidence?: number}> {
  // Implementation based on: {userDescription}
  // Use stockData[0] for analysis, but return PERCENTAGE CHANGES not absolute prices
  const predictions = [];
  const startDate = new Date();
  
  for (let i = 1; i <= days; i++) {
    const futureDate = new Date(startDate);
    futureDate.setDate(startDate.getDate() + i);
    
    // Return percentage multiplier (1.01 = +1%, 0.99 = -1%)
    const percentChange = 1 + (Math.random() - 0.5) * 0.02; // ±1% random
    
    predictions.push({
      date: futureDate.toISOString().split('T')[0],
      price: percentChange, // This is a multiplier, not absolute price!
      confidence: 0.8
    });
  }
  
  return predictions;
}

Generate the generatePredictions function now:
`;

function createRigidPrompt(userDescription: string, mode: 'forecast' | 'backtest'): string {
  const template = mode === 'forecast' ? FORECAST_PROMPT_TEMPLATE : BACKTEST_PROMPT_TEMPLATE;
  return template
    .replace(/{userDescription}/g, userDescription)
    .trim();
}

function extractTypeScriptCode(response: string, mode: 'forecast' | 'backtest'): string {
  let cleaned = response.trim();
  
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

function stripTypeScriptTypes(code: string): string {
  let jsCode = code;
  
  // Step 1: Remove return type annotations more carefully
  jsCode = jsCode.replace(/\):\s*Array<[^>]*>\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*number\[\]\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*string\[\]\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*any\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*number\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*string\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*boolean\s*\{/g, ') {');
  jsCode = jsCode.replace(/\):\s*void\s*\{/g, ') {');
  
  // Step 2: Remove parameter type annotations more carefully
  jsCode = jsCode.replace(/function\s+(\w+)\s*\([^)]*\)/g, (match) => {
    let cleanedMatch = match;
    
    // Remove parameter types: param: type -> param
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

function validateAndFixCode(code: string, mode: 'forecast' | 'backtest'): ValidationResult {
  const errors: string[] = [];
  let fixedCode = code;
  
  const functionName = mode === 'forecast' ? 'generatePredictions' : 'calculateWeights';
  const fallbackFunction = mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;
  
  if (!code.includes(functionName)) {
    errors.push(`Function name must be ${functionName}`);
    return { isValid: false, errors };
  }
  
  if (!code.includes('stockData: any[]') && !code.includes('stockData:any[]')) {
    if (code.includes('stockData')) {
      fixedCode = fixedCode.replace(/stockData:\s*\w+(\[\])?/g, 'stockData: any[]');
    } else {
      errors.push('Missing stockData parameter');
    }
  }
  
  // Check for dashboardParams parameter
  if (!code.includes('dashboardParams') && !code.includes('dashboardParams')) {
    // Add dashboardParams parameter if missing
    if (code.includes('stockData: any[]')) {
      fixedCode = fixedCode.replace(/stockData: any\[\]/, 'stockData: any[], dashboardParams: any = {}');
    }
  }
  
  if (mode === 'backtest') {
    // Validate backtest function
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
    // Validate forecast function
    if (!code.includes(': Array<')) {
      // Try to fix return type
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
  
  // Skip syntax validation for now to avoid corrupting valid TypeScript code
  // The type stripping will happen later during execution
  
  // try {
  //   // Strip TypeScript types before syntax validation
  //   const jsCode = stripTypeScriptTypes(fixedCode);
  //   new Function('return ' + jsCode);
  // } catch (syntaxError) {
  //   errors.push(`Syntax error: ${syntaxError}`);
  //   return { isValid: false, errors, fixedCode: fallbackFunction };
  // }
  
  return { isValid: errors.length === 0, errors, fixedCode };
}

function executeWithTimeout(code: string, stockData: StockData[], mode: 'forecast' | 'backtest', forecastDays: number = 30, timeoutMs: number = 3000, dashboardParams?: any): Promise<ExecutionResult> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      resolve({ success: false, error: 'Execution timeout', timeout: true });
    }, timeoutMs);
    
    try {
      let wrappedCode: string;
      let functionCall: string;
      
      let result: any;
      
      if (mode === 'forecast') {
        wrappedCode = `
          ${code}
          return generatePredictions(stockData, ${forecastDays}, dashboardParams);
        `;
        functionCall = 'generatePredictions';
        const fn = new Function('stockData', 'dashboardParams', wrappedCode);
        result = fn(stockData, dashboardParams || {});
      } else {
        wrappedCode = `
          ${code}
          return calculateWeights(stockData, dashboardParams);
        `;
        functionCall = 'calculateWeights';
        const fn = new Function('stockData', 'dashboardParams', wrappedCode);
        result = fn(stockData, dashboardParams || {});
      }
      
      clearTimeout(timeout);
      
      if (mode === 'forecast') {
        // Validate forecast result
        if (!Array.isArray(result)) {
          resolve({ success: false, error: 'Forecast function did not return an array', timeout: false });
          return;
        }
        
        // Validate prediction structure - keep multipliers as-is for frontend processing
        const predictions = result.map((p, index) => {
          if (!p || typeof p !== 'object') return null;
          const validatedPrediction = {
            date: p.date || new Date().toISOString().split('T')[0],
            price: typeof p.price === 'number' && isFinite(p.price) ? p.price : 1.0,
            confidence: typeof p.confidence === 'number' ? Math.max(0, Math.min(1, p.confidence)) : 0.8
          };
          
          return validatedPrediction;
        }).filter(p => p !== null);
        
        if (predictions.length === 0) {
          resolve({ success: false, error: 'No valid predictions generated', timeout: false });
          return;
        }
        
        resolve({ success: true, result: predictions as any, timeout: false });
        
      } else {
        // Validate backtest result (weights)
        if (!Array.isArray(result)) {
          resolve({ success: false, error: 'Function did not return an array', timeout: false });
          return;
        }
        
        const weights = result.map(w => {
          const num = Number(w);
          return isNaN(num) || !isFinite(num) ? 0 : num;
        });
        
        if (weights.length !== stockData.length) {
          const correctLength = stockData.length;
          const adjustedWeights = new Array(correctLength).fill(1.0 / correctLength);
          resolve({ success: true, result: adjustedWeights, timeout: false });
          return;
        }
        
        const sum = weights.reduce((a, b) => a + b, 0);
        const normalizedWeights = sum > 0 ? weights.map(w => w / sum) : new Array(weights.length).fill(1.0 / weights.length);
        
        resolve({ success: true, result: normalizedWeights, timeout: false });
      }
      
    } catch (error) {
      clearTimeout(timeout);
      resolve({ success: false, error: error?.toString() || 'Unknown execution error', timeout: false });
    }
  });
}

function generateFallbackResult(stockData: StockData[], layer: number, mode: 'forecast' | 'backtest', forecastDays: number = 30): number[] | any[] {
  if (mode === 'forecast') {
    // Generate fallback predictions
    const predictions = [];
    const startDate = new Date();
    const basePrice = stockData[0]?.price || 100;
    
    for (let i = 1; i <= forecastDays; i++) {
      const futureDate = new Date(startDate);
      futureDate.setDate(startDate.getDate() + i);
      
      let trend: number;
      switch (layer) {
        case 1: // Simple upward trend
          trend = 1.001; // 0.1% daily growth
          break;
        case 2: // Volatile trend
          trend = 1 + (Math.random() - 0.5) * 0.04; // ±2% daily change
          break;
        case 3: // Sideways trend
          trend = 1 + (Math.random() - 0.5) * 0.01; // ±0.5% daily change
          break;
        case 4: // Downward trend
          trend = 0.999; // -0.1% daily decline
          break;
        default: // Random walk
          trend = 1 + (Math.random() - 0.5) * 0.02; // ±1% daily change
      }
      
      const predictedPrice = basePrice * Math.pow(trend, i);
      
      predictions.push({
        date: futureDate.toISOString().split('T')[0],
        price: Math.round(predictedPrice * 100) / 100,
        confidence: Math.max(0.3, 0.9 - (i * 0.02))
      });
    }
    
    return predictions;
  } else {
    // Generate fallback weights for backtest
    const count = stockData.length || 1;
    
    switch (layer) {
      case 1: // Equal weights
        return new Array(count).fill(1.0 / count);
        
      case 2: // Market cap weighted (if available)
        const marketCaps = stockData.map(s => s.marketCap || 1);
        const totalCap = marketCaps.reduce((a, b) => a + b, 0);
        return totalCap > 0 ? marketCaps.map(cap => cap / totalCap) : new Array(count).fill(1.0 / count);
        
      case 3: // Price weighted
        const prices = stockData.map(s => s.price || 1);
        const totalPrice = prices.reduce((a, b) => a + b, 0);
        return totalPrice > 0 ? prices.map(price => price / totalPrice) : new Array(count).fill(1.0 / count);
        
      case 4: // Volume weighted (if available)
        const volumes = stockData.map(s => s.volume || 1);
        const totalVolume = volumes.reduce((a, b) => a + b, 0);
        return totalVolume > 0 ? volumes.map(vol => vol / totalVolume) : new Array(count).fill(1.0 / count);
        
      case 5: // Random weights (normalized)
        const randomWeights = new Array(count).fill(0).map(() => Math.random());
        const randomSum = randomWeights.reduce((a, b) => a + b, 0);
        return randomSum > 0 ? randomWeights.map(w => w / randomSum) : new Array(count).fill(1.0 / count);
        
      default: // Ultimate fallback
        return new Array(count).fill(1.0 / count);
    }
  }
}

export async function generatePortfolioWeights(
  userDescription: string, 
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  claudeApiCall?: (prompt: string) => Promise<string>,
  securityConfig?: SecurityConfig,
  forecastDays: number = 30,
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  }
): Promise<GenerationResult> {
  const startTime = Date.now();
  
  if (!stockData || stockData.length === 0) {
    const fallbackResult = generateFallbackResult([{ symbol: 'DEFAULT', price: 100 }], 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: 'Empty stock data',
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }
  
  // Layer 0: Security validation of user prompt
  const promptSecurity = validateSecurity(userDescription, undefined, securityConfig);
  if (!promptSecurity.overallValid) {
    const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: `Security validation failed: ${promptSecurity.promptValidation.blockedReason}`,
      fallbackUsed: true,
      executionTime: Date.now() - startTime,
      securityValidation: {
        promptValid: false,
        blockedReason: promptSecurity.promptValidation.blockedReason,
        riskLevel: promptSecurity.combinedRiskLevel
      }
    };
  }
  
  // Layer 1: Try Claude API generation
  if (claudeApiCall) {
    try {
      const prompt = createRigidPrompt(userDescription, mode);
      const response = await claudeApiCall(prompt);
      
      // Layer 2: Extract and clean response
      const extractedCode = extractTypeScriptCode(response, mode);
      
      // Layer 2.5: Security validation of generated code
      const codeSecurity = validateSecurity(userDescription, extractedCode, securityConfig);
      if (!codeSecurity.overallValid) {
        // Use fallback instead of failing completely
        const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
        return {
          success: true,
          type: mode,
          ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
          error: `Generated code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime,
          securityValidation: {
            promptValid: true,
            codeValid: false,
            blockedReason: codeSecurity.codeValidation?.blockedReason,
            riskLevel: codeSecurity.combinedRiskLevel
          }
        };
      }
      
      // Layer 3: Validate and fix code
      const validation = validateAndFixCode(extractedCode, mode);
      let codeToExecute = validation.fixedCode || extractedCode;
      
      // Strip TypeScript types for JavaScript execution
      codeToExecute = stripTypeScriptTypes(codeToExecute);
      
      // Layer 4: Execute with timeout
      const execution = await executeWithTimeout(codeToExecute, stockData, mode, forecastDays, 3000, dashboardParams);
      
      if (execution.success && execution.result) {
        return {
          success: true,
          type: mode,
          ...(mode === 'backtest' ? { weights: execution.result as number[] } : { predictions: execution.result as any[] }),
          code: codeToExecute,
          fallbackUsed: false,
          executionTime: Date.now() - startTime,
          securityValidation: {
            promptValid: true,
            codeValid: true,
            riskLevel: 'none'
          }
        };
      }
    } catch (error) {
      // Fall through to fallback layers
    }
  }
  
  // Layer 5-9: Multiple fallback strategies
  for (let layer = 1; layer <= 5; layer++) {
    try {
      const fallbackResult = generateFallbackResult(stockData, layer, mode, forecastDays);
      
      if (mode === 'backtest' && (fallbackResult as number[]).length === stockData.length) {
        return {
          success: true,
          type: mode,
          weights: fallbackResult as number[],
          error: `Used fallback layer ${layer}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime
        };
      } else if (mode === 'forecast' && Array.isArray(fallbackResult) && fallbackResult.length > 0) {
        return {
          success: true,
          type: mode,
          predictions: fallbackResult as any[],
          error: `Used fallback layer ${layer}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime
        };
      }
    } catch (error) {
      continue;
    }
  }
  
  // Ultimate fallback (Layer 10)
  const ultimateFallback = generateFallbackResult(stockData, 1, mode, forecastDays);
  return {
    success: true,
    type: mode,
    ...(mode === 'backtest' ? { weights: ultimateFallback as number[] } : { predictions: ultimateFallback as any[] }),
    error: 'Used ultimate fallback',
    fallbackUsed: true,
    executionTime: Date.now() - startTime
  };
}

// New function: Generate code only (no execution)
export async function generateCodeOnly(
  userDescription: string, 
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  claudeApiCall?: (prompt: string) => Promise<string>,
  securityConfig?: SecurityConfig,
  forecastDays: number = 30,
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  }
): Promise<{ success: boolean; code?: string; error?: string; securityValidation?: any }> {
  const startTime = Date.now();
  
  if (!stockData || stockData.length === 0) {
    return {
      success: false,
      error: 'Empty stock data'
    };
  }
  
  // Security validation of user prompt
  const promptSecurity = validateSecurity(userDescription, undefined, securityConfig);
  if (!promptSecurity.overallValid) {
    return {
      success: false,
      error: `Security validation failed: ${promptSecurity.promptValidation.blockedReason}`,
      securityValidation: {
        promptValid: false,
        blockedReason: promptSecurity.promptValidation.blockedReason,
        riskLevel: promptSecurity.combinedRiskLevel
      }
    };
  }
  
  // Try Claude API generation
  if (claudeApiCall) {
    try {
      const prompt = createRigidPrompt(userDescription, mode);
      const response = await claudeApiCall(prompt);
      
      // Extract and clean response
      const extractedCode = extractTypeScriptCode(response, mode);
      
      // Security validation of generated code
      const codeSecurity = validateSecurity(userDescription, extractedCode, securityConfig);
      if (!codeSecurity.overallValid) {
        return {
          success: false,
          error: `Generated code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
          securityValidation: {
            promptValid: true,
            codeValid: false,
            blockedReason: codeSecurity.codeValidation?.blockedReason,
            riskLevel: codeSecurity.combinedRiskLevel
          }
        };
      }
      
      // Validate and fix code
      const validation = validateAndFixCode(extractedCode, mode);
      let codeToReturn = validation.fixedCode || extractedCode;
      
      return {
        success: true,
        code: codeToReturn,
        securityValidation: {
          promptValid: true,
          codeValid: true,
          riskLevel: 'none'
        }
      };
    } catch (error) {
      return {
        success: false,
        error: `Code generation failed: ${error}`
      };
    }
  }
  
  // Fallback - return template code
  const fallbackFunction = mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;
  return {
    success: true,
    code: fallbackFunction,
    error: 'Used fallback template'
  };
}

// New function: Execute user-provided code
export async function executeUserCode(
  code: string,
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  forecastDays: number = 30,
  dashboardParams?: any,
  securityConfig?: SecurityConfig
): Promise<GenerationResult> {
  const startTime = Date.now();
  
  if (!stockData || stockData.length === 0) {
    const fallbackResult = generateFallbackResult([{ symbol: 'DEFAULT', price: 100 }], 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: 'Empty stock data',
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }
  
  try {
    // Security validation of code before execution
    const codeSecurity = validateSecurity('', code, securityConfig);
    if (!codeSecurity.overallValid) {
      const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
      return {
        success: false,
        type: mode,
        ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
        error: `Code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
        fallbackUsed: true,
        executionTime: Date.now() - startTime,
        securityValidation: {
          promptValid: true,
          codeValid: false,
          blockedReason: codeSecurity.codeValidation?.blockedReason,
          riskLevel: codeSecurity.combinedRiskLevel
        }
      };
    }
    
    // Validate and fix code
    const validation = validateAndFixCode(code, mode);
    let codeToExecute = validation.fixedCode || code;
    
    // Strip TypeScript types for JavaScript execution
    codeToExecute = stripTypeScriptTypes(codeToExecute);
    
    // Execute with timeout
    const execution = await executeWithTimeout(codeToExecute, stockData, mode, forecastDays, 3000, dashboardParams);
    
    if (execution.success && execution.result) {
      return {
        success: true,
        type: mode,
        ...(mode === 'backtest' ? { weights: execution.result as number[] } : { predictions: execution.result as any[] }),
        code: codeToExecute,
        fallbackUsed: false,
        executionTime: Date.now() - startTime,
        securityValidation: {
          promptValid: true,
          codeValid: true,
          riskLevel: 'none'
        }
      };
    } else {
      // Execution failed, use fallback
      const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
      return {
        success: true,
        type: mode,
        ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
        error: execution.error || 'Execution failed',
        fallbackUsed: true,
        executionTime: Date.now() - startTime
      };
    }
  } catch (error) {
    // Execution error, use fallback
    const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
    return {
      success: true,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: `Execution error: ${error}`,
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }
}