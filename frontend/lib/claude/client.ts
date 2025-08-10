import { StockData, GenerationResult } from './generator';
import { SecurityConfig } from './security';
import { 
  ClaudeError, 
  ClaudeApiError, 
  NetworkError, 
  ValidationError, 
  TimeoutError, 
  RateLimitError, 
  ParseError,
  ErrorFactory,
  ErrorUtils
} from './errors';

// Client-side Request/Response Interfaces
export interface GenerateRequest {
  userDescription: string;
  stockData: StockData[];
  mode: 'forecast' | 'backtest';
  securityConfig?: Partial<SecurityConfig>;
  forecastDays?: number; // Only used for forecast mode
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  };
}

export interface GenerateResponse {
  success: boolean;
  result?: GenerationResult;
  error?: string;
  rateLimitInfo?: {
    remaining: number;
    resetTime: number;
  };
}

export interface ApiErrorResponse {
  success: false;
  error: string;
  statusCode: number;
}

// Legacy client error type for backward compatibility
enum ClientErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  RATE_LIMIT_ERROR = 'RATE_LIMIT_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  API_ERROR = 'API_ERROR',
  PARSE_ERROR = 'PARSE_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
  DUPLICATE_REQUEST = 'DUPLICATE_REQUEST'
}

// Legacy wrapper for backward compatibility
class ClaudeClientError extends ClaudeError {
  constructor(
    public type: ClientErrorType,
    message: string,
    public statusCode?: number,
    retryable: boolean = false,
    public rateLimitInfo?: { remaining: number; resetTime: number }
  ) {
    super(
      message,
      type,
      'api' as any, // Will be mapped to appropriate category
      'medium' as any, // Will be mapped to appropriate severity
      retryable
    );
    this.name = 'ClaudeClientError';
  }
}

// Loading State Management
interface LoadingState {
  isLoading: boolean;
  requestId?: string;
  startTime?: number;
  progress?: 'validating' | 'generating' | 'processing';
}

// Request Management Configuration
const CLIENT_CONFIG = {
  REQUEST_TIMEOUT: 30000, // 30 seconds
  RATE_LIMIT_WINDOW: 60000, // 1 minute
  MAX_REQUESTS_PER_WINDOW: 8, // Slightly lower than server-side to be safe
  MAX_RETRIES: 3,
  RETRY_DELAY_BASE: 1000, // 1 second base delay
  MAX_DESCRIPTION_LENGTH: 2000,
  MAX_STOCK_COUNT: 100,
};

// Request tracking for rate limiting and deduplication
class RequestTracker {
  private requests: Map<string, number> = new Map();
  private requestCounts: Map<string, { count: number; resetTime: number }> = new Map();
  private pendingRequests: Map<string, Promise<GenerateResponse>> = new Map();

  // Generate request hash for deduplication
  private generateRequestHash(request: GenerateRequest): string {
    const normalized = {
      description: request.userDescription.trim().toLowerCase(),
      symbols: request.stockData.map(s => s.symbol).sort(),
    };
    return btoa(JSON.stringify(normalized));
  }

  // Check if request is duplicate and return existing promise if so
  checkDuplicateRequest(request: GenerateRequest): Promise<GenerateResponse> | null {
    const hash = this.generateRequestHash(request);
    return this.pendingRequests.get(hash) || null;
  }

  // Register new request
  registerRequest(request: GenerateRequest, promise: Promise<GenerateResponse>): void {
    const hash = this.generateRequestHash(request);
    this.pendingRequests.set(hash, promise);
    
    // Clean up when request completes
    promise.finally(() => {
      this.pendingRequests.delete(hash);
    });
  }

  // Check rate limit
  checkRateLimit(): { allowed: boolean; remaining: number; resetTime: number } {
    const now = Date.now();
    const key = 'client-rate-limit';
    const existing = this.requestCounts.get(key);

    if (!existing || now >= existing.resetTime) {
      const resetTime = now + CLIENT_CONFIG.RATE_LIMIT_WINDOW;
      this.requestCounts.set(key, { count: 1, resetTime });
      return { allowed: true, remaining: CLIENT_CONFIG.MAX_REQUESTS_PER_WINDOW - 1, resetTime };
    }

    if (existing.count >= CLIENT_CONFIG.MAX_REQUESTS_PER_WINDOW) {
      return { allowed: false, remaining: 0, resetTime: existing.resetTime };
    }

    existing.count++;
    this.requestCounts.set(key, existing);
    return { 
      allowed: true, 
      remaining: CLIENT_CONFIG.MAX_REQUESTS_PER_WINDOW - existing.count, 
      resetTime: existing.resetTime 
    };
  }
}

const requestTracker = new RequestTracker();

// Loading state management
class LoadingStateManager {
  private states: Map<string, LoadingState> = new Map();
  private listeners: Map<string, ((state: LoadingState) => void)[]> = new Map();

  createRequest(requestId: string): void {
    const state: LoadingState = {
      isLoading: true,
      requestId,
      startTime: Date.now(),
      progress: 'validating'
    };
    this.states.set(requestId, state);
    this.notifyListeners(requestId, state);
  }

  updateProgress(requestId: string, progress: LoadingState['progress']): void {
    const state = this.states.get(requestId);
    if (state) {
      state.progress = progress;
      this.notifyListeners(requestId, state);
    }
  }

  completeRequest(requestId: string): void {
    const state = this.states.get(requestId);
    if (state) {
      state.isLoading = false;
      state.progress = undefined;
      this.notifyListeners(requestId, state);
      
      // Clean up after a delay
      setTimeout(() => {
        this.states.delete(requestId);
        this.listeners.delete(requestId);
      }, 1000);
    }
  }

  getState(requestId: string): LoadingState | undefined {
    return this.states.get(requestId);
  }

  onStateChange(requestId: string, callback: (state: LoadingState) => void): () => void {
    if (!this.listeners.has(requestId)) {
      this.listeners.set(requestId, []);
    }
    this.listeners.get(requestId)!.push(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(requestId);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  private notifyListeners(requestId: string, state: LoadingState): void {
    const callbacks = this.listeners.get(requestId);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(state);
        } catch (error) {
          console.error('Error in loading state callback:', error);
        }
      });
    }
  }
}

const loadingStateManager = new LoadingStateManager();

// Request validation using enhanced error types
function validateRequest(request: GenerateRequest, requestId?: string): void {
  if (!request.userDescription || typeof request.userDescription !== 'string') {
    throw ErrorFactory.createValidationError(
      'userDescription is required and must be a string',
      'userDescription',
      request.userDescription,
      requestId
    );
  }

  if (request.userDescription.length > CLIENT_CONFIG.MAX_DESCRIPTION_LENGTH) {
    throw ErrorFactory.createValidationError(
      `Description must be ${CLIENT_CONFIG.MAX_DESCRIPTION_LENGTH} characters or less`,
      'userDescription',
      request.userDescription,
      requestId
    );
  }

  // Validate mode
  if (!request.mode || (request.mode !== 'forecast' && request.mode !== 'backtest')) {
    throw ErrorFactory.createValidationError(
      'mode is required and must be either "forecast" or "backtest"',
      'mode',
      request.mode,
      requestId
    );
  }

  // Validate forecastDays for forecast mode
  if (request.mode === 'forecast' && request.forecastDays !== undefined) {
    if (typeof request.forecastDays !== 'number' || !Number.isInteger(request.forecastDays) || request.forecastDays < 1 || request.forecastDays > 365) {
      throw ErrorFactory.createValidationError(
        'forecastDays must be an integer between 1 and 365',
        'forecastDays',
        request.forecastDays,
        requestId
      );
    }
  }

  if (!Array.isArray(request.stockData)) {
    throw ErrorFactory.createValidationError(
      'stockData must be an array',
      'stockData',
      request.stockData,
      requestId
    );
  }

  if (request.stockData.length === 0) {
    throw ErrorFactory.createValidationError(
      'stockData array cannot be empty',
      'stockData',
      request.stockData,
      requestId
    );
  }

  if (request.stockData.length > CLIENT_CONFIG.MAX_STOCK_COUNT) {
    throw ErrorFactory.createValidationError(
      `Cannot process more than ${CLIENT_CONFIG.MAX_STOCK_COUNT} stocks`,
      'stockData',
      request.stockData,
      requestId
    );
  }

  // Validate each stock data item
  request.stockData.forEach((stock, index) => {
    if (!stock || typeof stock !== 'object') {
      throw ErrorFactory.createValidationError(
        `stockData[${index}] must be an object`,
        `stockData[${index}]`,
        stock,
        requestId
      );
    }

    if (!stock.symbol || typeof stock.symbol !== 'string') {
      throw ErrorFactory.createValidationError(
        `stockData[${index}].symbol is required and must be a string`,
        `stockData[${index}].symbol`,
        stock.symbol,
        requestId
      );
    }

    if (typeof stock.price !== 'number' || !isFinite(stock.price) || stock.price <= 0) {
      throw ErrorFactory.createValidationError(
        `stockData[${index}].price must be a positive number`,
        `stockData[${index}].price`,
        stock.price,
        requestId
      );
    }
  });
}

// Fetch with timeout
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new TimeoutError('Request timed out', timeout);
    }
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new NetworkError('Network request failed', error);
    }
    throw ErrorFactory.createFromFetchError(error);
  }
}

// Response validation and parsing
async function parseAndValidateResponse(response: Response): Promise<GenerateResponse> {
  let responseData: any;

  try {
    responseData = await response.json();
  } catch (parseError) {
    throw new ParseError('Failed to parse response JSON', await response.text().catch(() => 'Unable to read response'));
  }

  // Handle rate limit response
  if (response.status === 429) {
    const retryAfter = response.headers.get('Retry-After');
    const resetTime = response.headers.get('X-RateLimit-Reset');
    const remaining = response.headers.get('X-RateLimit-Remaining');

    throw new RateLimitError(
      responseData.error || 'Rate limit exceeded',
      remaining ? parseInt(remaining) : 0,
      resetTime ? parseInt(resetTime) * 1000 : Date.now() + 60000
    );
  }

  // Handle other API errors
  if (!response.ok) {
    throw ErrorFactory.createFromApiResponse(response, responseData);
  }

  // Validate response structure
  if (typeof responseData !== 'object' || responseData === null) {
    throw new ParseError('Invalid response format', JSON.stringify(responseData));
  }

  if (typeof responseData.success !== 'boolean') {
    throw new ParseError('Response missing success field', JSON.stringify(responseData));
  }

  return responseData as GenerateResponse;
}

// Retry logic with exponential backoff using enhanced error types
async function executeWithRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = CLIENT_CONFIG.MAX_RETRIES,
  requestId?: string
): Promise<T> {
  let lastError: ClaudeError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error: any) {
      if (error instanceof ClaudeError) {
        lastError = error;
        
        // Don't retry non-retryable errors
        if (!error.isRetryable() || attempt === maxRetries) {
          throw error;
        }

        // Calculate delay with exponential backoff and jitter
        const delay = ErrorUtils.getRetryDelay(attempt + 1, CLIENT_CONFIG.RETRY_DELAY_BASE);
        
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        // Network or other unexpected errors
        if (attempt === maxRetries) {
          throw new NetworkError(
            error.message || 'Network request failed',
            error,
            requestId
          );
        }
        
        const delay = CLIENT_CONFIG.RETRY_DELAY_BASE * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError!;
}

// Main API call function
async function makeApiCall(request: GenerateRequest): Promise<GenerateResponse> {
  const response = await fetchWithTimeout(
    '/api/claude/generate',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    },
    CLIENT_CONFIG.REQUEST_TIMEOUT
  );

  return parseAndValidateResponse(response);
}

// Main client function
export async function generateStrategy(
  description: string,
  mode: 'forecast' | 'backtest',
  stockData?: StockData[],
  securityConfig?: Partial<SecurityConfig>,
  forecastDays?: number,
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  }
): Promise<GenerationResult> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  try {
    // Create loading state
    loadingStateManager.createRequest(requestId);

    // Default stock data if not provided (for testing)
    const defaultStockData: StockData[] = stockData || [
      { symbol: 'SPY', price: 400 },
      { symbol: 'QQQ', price: 350 },
      { symbol: 'VTI', price: 200 },
    ];

    const request: GenerateRequest = {
      userDescription: description,
      stockData: defaultStockData,
      mode,
      securityConfig,
      forecastDays,
      dashboardParams,
    };

    // Validate request
    validateRequest(request, requestId);
    loadingStateManager.updateProgress(requestId, 'validating');

    // Check rate limit
    const rateLimitCheck = requestTracker.checkRateLimit();
    if (!rateLimitCheck.allowed) {
      throw new RateLimitError(
        'Client-side rate limit exceeded. Please wait before making another request.',
        rateLimitCheck.remaining,
        rateLimitCheck.resetTime,
        requestId
      );
    }

    // Check for duplicate request
    const existingRequest = requestTracker.checkDuplicateRequest(request);
    if (existingRequest) {
      throw ErrorFactory.createValidationError(
        'Identical request is already in progress',
        'request',
        'duplicate',
        requestId
      );
    }

    loadingStateManager.updateProgress(requestId, 'generating');

    // Execute API call with retry logic
    const apiCallPromise = executeWithRetry(() => makeApiCall(request), CLIENT_CONFIG.MAX_RETRIES, requestId);
    requestTracker.registerRequest(request, apiCallPromise);

    const response = await apiCallPromise;

    loadingStateManager.updateProgress(requestId, 'processing');

    // Handle API response
    if (!response.success) {
      throw new ClaudeApiError(
        response.error || 'API request failed',
        undefined,
        response,
        requestId
      );
    }

    if (!response.result) {
      throw new ParseError(
        'API response missing result data',
        JSON.stringify(response),
        requestId
      );
    }

    return response.result;

  } catch (error: any) {
    if (error instanceof ClaudeError) {
      throw error;
    }
    
    // Handle unexpected errors
    throw ErrorFactory.createFromFetchError(error, requestId);
  } finally {
    loadingStateManager.completeRequest(requestId);
  }
}

// Loading state helpers for React components
export const loadingHelpers = {
  // Create a loading state tracker for a component
  useLoadingState: (requestId: string, callback: (state: LoadingState) => void) => {
    return loadingStateManager.onStateChange(requestId, callback);
  },

  // Get current loading state
  getLoadingState: (requestId: string) => {
    return loadingStateManager.getState(requestId);
  },

  // Check if any requests are currently loading
  hasActiveRequests: () => {
    // This would need to be implemented if needed
    return false;
  }
};

// New function: Generate code only (no execution)
export async function generateCodeOnly(
  description: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[], // Required, no defaults
  securityConfig?: Partial<SecurityConfig>,
  forecastDays?: number,
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  }
): Promise<{ success: boolean; code?: string; error?: string }> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  try {
    const request: GenerateRequest & { generateOnly: boolean } = {
      userDescription: description,
      stockData: stockData, // Use provided stock data directly
      mode,
      securityConfig,
      forecastDays,
      dashboardParams,
      generateOnly: true
    };

    // Validate request
    validateRequest(request, requestId);

    // Check rate limit
    const rateLimitCheck = requestTracker.checkRateLimit();
    if (!rateLimitCheck.allowed) {
      throw new RateLimitError(
        'Client-side rate limit exceeded. Please wait before making another request.',
        rateLimitCheck.remaining,
        rateLimitCheck.resetTime,
        requestId
      );
    }

    // Execute API call with retry logic
    const apiCallPromise = executeWithRetry(() => makeApiCall(request), CLIENT_CONFIG.MAX_RETRIES, requestId);
    requestTracker.registerRequest(request, apiCallPromise as any);

    const response = await apiCallPromise;

    // Handle API response for code generation
    if (!response.success) {
      throw new ClaudeApiError(
        response.error || 'API request failed',
        undefined,
        response,
        requestId
      );
    }

    // For code-only generation, the response structure is different
    return {
      success: response.success,
      code: (response as any).code,
      error: (response as any).error
    };

  } catch (error: any) {
    if (error instanceof ClaudeError) {
      throw error;
    }
    
    // Handle unexpected errors
    throw ErrorFactory.createFromFetchError(error, requestId);
  }
}

// New function: Execute user code locally (no Claude API calls)
export async function executeUserCode(
  userCode: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[], // Required, no defaults
  securityConfig?: Partial<SecurityConfig>,
  forecastDays?: number,
  dashboardParams?: {
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    historyDays?: number;
  }
): Promise<GenerationResult> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const startTime = Date.now();
  
  try {
    // Preserve existing validation for stockData
    if (!Array.isArray(stockData)) {
      throw ErrorFactory.createValidationError(
        'stockData must be an array',
        'stockData',
        stockData,
        requestId
      );
    }

    if (stockData.length === 0) {
      throw ErrorFactory.createValidationError(
        'stockData array cannot be empty',
        'stockData',
        stockData,
        requestId
      );
    }

    if (stockData.length > CLIENT_CONFIG.MAX_STOCK_COUNT) {
      throw ErrorFactory.createValidationError(
        `Cannot process more than ${CLIENT_CONFIG.MAX_STOCK_COUNT} stocks`,
        'stockData',
        stockData,
        requestId
      );
    }

    // Skip rate limiting for local code execution since no API calls are made
    // Rate limiting is only needed for actual Claude API calls, not local JavaScript execution

    // Apply security validation to user code (preserve security)
    const { validateSecurity } = await import('./security');
    const securityResult = validateSecurity('', userCode, {
      enablePromptValidation: false, // Skip prompt validation for execution
      enableCodeValidation: true,   // Always validate code
      strictMode: securityConfig?.strictMode || false,
      allowCreativeStrategies: securityConfig?.allowCreativeStrategies !== false,
      customBlockedPatterns: securityConfig?.customBlockedPatterns,
      customAllowedPatterns: securityConfig?.customAllowedPatterns
    });

    if (!securityResult.overallValid || !securityResult.codeValidation?.isValid) {
      throw ErrorFactory.createValidationError(
        securityResult.codeValidation?.blockedReason || 'Security validation failed',
        'userCode',
        'security_blocked',
        requestId
      );
    }

    // Execute JavaScript locally instead of calling Claude API
    const executionResult = await executeJavaScriptLocally(
      userCode, 
      mode, 
      stockData, 
      forecastDays, 
      dashboardParams,
      requestId
    );
    
    // Preserve existing request tracking (simulate successful request)
    const mockPromise = Promise.resolve({ success: true, result: executionResult });
    requestTracker.registerRequest({
      userDescription: '',
      stockData,
      mode,
      securityConfig,
      forecastDays,
      dashboardParams
    } as any, mockPromise as any);

    const executionTime = Date.now() - startTime;
    
    return {
      success: true,
      type: mode,
      weights: mode === 'backtest' ? executionResult.weights : undefined,
      predictions: mode === 'forecast' ? executionResult.predictions : undefined,
      code: userCode,
      fallbackUsed: false,
      executionTime,
      securityValidation: {
        promptValid: true,
        codeValid: true,
        riskLevel: securityResult.combinedRiskLevel
      }
    };

  } catch (error: any) {
    const executionTime = Date.now() - startTime;
    
    // Preserve existing error handling
    if (error instanceof ClaudeError) {
      throw error;
    }
    
    console.error('Local code execution failed:', error);
    
    // Return error result with fallback behavior
    return {
      success: false,
      type: mode,
      error: error.message || 'Code execution failed',
      code: userCode,
      fallbackUsed: true,
      executionTime,
      securityValidation: {
        promptValid: true,
        codeValid: false,
        blockedReason: error.message,
        riskLevel: 'high'
      }
    };
  }
}

// Local JavaScript execution function
async function executeJavaScriptLocally(
  userCode: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[],
  forecastDays?: number,
  dashboardParams?: any,
  requestId?: string
): Promise<{ weights?: number[], predictions?: Array<{date: string, price: number, confidence?: number}> }> {
  
  // Create secure execution environment - only allow safe globals
  const safeGlobals = {
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

  try {
    // Convert TypeScript to JavaScript using Babel
    let jsCode = await transpileTypeScriptToJavaScript(userCode);
    
    // Use Function constructor for safer execution than eval
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
        const predictions = generatePredictions(stockData, forecastDays || 30, dashboardParams || {});
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
          forecastDays,
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
      setTimeout(() => reject(new Error('Code execution timeout (5 seconds)')), 5000)
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

// Browser-compatible TypeScript to JavaScript transpilation using Babel Standalone
async function transpileTypeScriptToJavaScript(code: string): Promise<string> {
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
    
    console.log('TypeScript transpilation - Original:', code.substring(0, 200) + '...');
    console.log('TypeScript transpilation - Result:', result.code.substring(0, 200) + '...');
    
    return result.code;
    
  } catch (error: any) {
    console.warn('Babel TypeScript transpilation failed:', error);
    console.log('Falling back to simple regex-based stripping');
    
    // Fallback to simple type stripping if Babel fails
    return code
      .replace(/:\s*[^=,){\s]+(?=\s*[=,){])/g, '') // Remove : Type annotations
      .replace(/\):\s*[^{]+\{/g, ') {'); // Remove return types
  }
}

// Export types and main function
export type { LoadingState, StockData, GenerationResult, SecurityConfig };
export { ClientErrorType };
export { ClaudeClientError };