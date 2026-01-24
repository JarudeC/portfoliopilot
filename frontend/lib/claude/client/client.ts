/**
 * Claude AI client for strategy generation.
 * Handles API requests to /api/claude/generate with rate limiting, request deduplication,
 * retry logic, and local JavaScript execution for user-provided code.
 *
 * Main entry points:
 * - generateCodeOnly(): Generate code without execution (for user review)
 * - executeUserCode(): Execute user-provided code locally with security validation
 */

import type { StockData, GenerationResult } from '../core/types';
import type { SecurityConfig } from '../execution/security';
import {
  API_REQUEST_TIMEOUT,
  DEFAULT_MAX_RETRIES,
  RETRY_DELAY_BASE,
} from '../core/constants';
import {
  ClaudeError,
  ClaudeApiError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  ParseError,
  ErrorFactory,
  ErrorUtils,
} from '../core/errors';
import { requestTracker } from './request-tracker';
import { executeJavaScriptLocally } from '../execution/code-sandbox';

// ============================================================================
// Client-side Request/Response Interfaces
// ============================================================================

export interface GenerateRequest {
  userDescription: string;
  stockData: StockData[];
  mode: 'forecast' | 'backtest';
  securityConfig?: Partial<SecurityConfig>;
  forecastDays?: number;
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

// ============================================================================
// Request Validation
// ============================================================================

/**
 * Validate request parameters before sending to API.
 * Minimal validation - UI already enforces most constraints.
 */
function validateRequest(request: GenerateRequest, requestId?: string): void {
  // Basic type checks for API protection
  if (!request.userDescription || typeof request.userDescription !== 'string') {
    throw ErrorFactory.createValidationError(
      'userDescription is required and must be a string',
      'userDescription',
      request.userDescription,
      requestId
    );
  }

  if (!Array.isArray(request.stockData) || request.stockData.length === 0) {
    throw ErrorFactory.createValidationError(
      'stockData must be a non-empty array',
      'stockData',
      request.stockData,
      requestId
    );
  }
}

// ============================================================================
// HTTP Layer
// ============================================================================

/**
 * Fetch with timeout protection.
 */
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

/**
 * Parse and validate API response.
 */
async function parseAndValidateResponse(response: Response): Promise<GenerateResponse> {
  let responseData: any;

  try {
    responseData = await response.json();
  } catch (parseError) {
    throw new ParseError('Failed to parse response JSON', await response.text().catch(() => 'Unable to read response'));
  }

  if (response.status === 429) {
    const resetTime = response.headers.get('X-RateLimit-Reset');
    const remaining = response.headers.get('X-RateLimit-Remaining');

    throw new RateLimitError(
      responseData.error || 'Rate limit exceeded',
      remaining ? parseInt(remaining) : 0,
      resetTime ? parseInt(resetTime) * 1000 : Date.now() + 60000
    );
  }

  if (!response.ok) {
    throw ErrorFactory.createFromApiResponse(response, responseData);
  }

  if (typeof responseData !== 'object' || responseData === null) {
    throw new ParseError('Invalid response format', JSON.stringify(responseData));
  }

  if (typeof responseData.success !== 'boolean') {
    throw new ParseError('Response missing success field', JSON.stringify(responseData));
  }

  return responseData as GenerateResponse;
}

// ============================================================================
// Retry Logic
// ============================================================================

/**
 * Execute operation with exponential backoff retry.
 */
async function executeWithRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = DEFAULT_MAX_RETRIES,
  requestId?: string
): Promise<T> {
  let lastError: ClaudeError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error: any) {
      if (error instanceof ClaudeError) {
        lastError = error;

        if (!error.isRetryable() || attempt === maxRetries) {
          throw error;
        }

        const delay = ErrorUtils.getRetryDelay(attempt + 1, RETRY_DELAY_BASE);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        if (attempt === maxRetries) {
          throw new NetworkError(
            error.message || 'Network request failed',
            error,
            requestId
          );
        }

        const delay = RETRY_DELAY_BASE * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError!;
}

// ============================================================================
// API Call
// ============================================================================

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
    API_REQUEST_TIMEOUT
  );

  return parseAndValidateResponse(response);
}

// ============================================================================
// Public API Functions
// ============================================================================

/**
 * Generate strategy code without executing it.
 * Execution happens later in useForecast/useBacktest with real data.
 */
export async function generateCodeOnly(
  description: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[],
  securityConfig?: Partial<SecurityConfig>,
  forecastDays?: number
): Promise<{ success: boolean; code?: string; error?: string }> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  try {
    const request: GenerateRequest & { generateOnly: boolean } = {
      userDescription: description,
      stockData: stockData,
      mode,
      securityConfig,
      forecastDays,
      generateOnly: true
    };

    validateRequest(request, requestId);

    const rateLimitCheck = requestTracker.checkRateLimit();
    if (!rateLimitCheck.allowed) {
      throw new RateLimitError(
        'Client-side rate limit exceeded. Please wait before making another request.',
        rateLimitCheck.remaining,
        rateLimitCheck.resetTime,
        requestId
      );
    }

    const apiCallPromise = executeWithRetry(() => makeApiCall(request), DEFAULT_MAX_RETRIES, requestId);
    requestTracker.registerRequest(request, apiCallPromise as any);

    const response = await apiCallPromise;

    if (!response.success) {
      throw new ClaudeApiError(
        response.error || 'API request failed',
        undefined,
        response,
        requestId
      );
    }

    return {
      success: response.success,
      code: (response as any).code,
      error: (response as any).error
    };

  } catch (error: any) {
    if (error instanceof ClaudeError) {
      throw error;
    }
    throw ErrorFactory.createFromFetchError(error, requestId);
  }
}

/**
 * Execute user-provided code locally without calling Claude API.
 */
export async function executeUserCode(
  userCode: string,
  mode: 'forecast' | 'backtest',
  stockData: StockData[],
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
    if (!Array.isArray(stockData) || stockData.length === 0) {
      throw ErrorFactory.createValidationError(
        'stockData must be a non-empty array',
        'stockData',
        stockData,
        requestId
      );
    }

    // Apply security validation to user code
    const { validateSecurity } = await import('../execution/security');
    const securityResult = validateSecurity('', userCode, {
      enablePromptValidation: false,
      enableCodeValidation: true,
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

    const executionResult = await executeJavaScriptLocally(
      userCode,
      mode,
      stockData,
      forecastDays,
      dashboardParams,
      requestId
    );

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

    if (error instanceof ClaudeError) {
      throw error;
    }

    console.error('Local code execution failed:', error);

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

// ============================================================================
// Exports
// ============================================================================

export type { StockData, GenerationResult, SecurityConfig };
