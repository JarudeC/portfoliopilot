import { NextRequest, NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { generatePortfolioWeights, generateCodeOnly, executeUserCode, StockData, GenerationResult } from '../../../../lib/claude/generator';
import { validateSecurity, createSecurityConfig, SecurityConfig } from '../../../../lib/claude/security';
import { 
  ClaudeError, 
  ClaudeApiError, 
  NetworkError, 
  ValidationError, 
  TimeoutError, 
  RateLimitError, 
  SecurityError,
  ErrorFactory,
  ErrorUtils
} from '../../../../lib/claude/errors';

// Request/Response Type Interfaces
interface GenerateRequest {
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
  // New parameters for code review feature
  generateOnly?: boolean; // If true, only generate code without execution
  userCode?: string; // User-provided code to execute
}

interface GenerateResponse {
  success: boolean;
  result?: GenerationResult;
  error?: string;
  rateLimitInfo?: {
    remaining: number;
    resetTime: number;
  };
}

interface ErrorResponse {
  success: false;
  error: string;
  statusCode: number;
  suggestions?: string[];
}

// Rate Limiting Configuration
const RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 10; // 10 requests per minute per IP
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_DESCRIPTION_LENGTH = 2000;
const MAX_STOCK_DATA_COUNT = 100;

// In-memory rate limiting store (use Redis in production)
const rateLimitStore = new Map<string, { count: number; resetTime: number }>();

// Utility Functions
function getClientIP(request: NextRequest): string {
  const forwarded = request.headers.get('x-forwarded-for');
  const realIP = request.headers.get('x-real-ip');
  
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  
  if (realIP) {
    return realIP;
  }
  
  return 'unknown';
}

function checkRateLimit(ip: string): { allowed: boolean; remaining: number; resetTime: number } {
  const now = Date.now();
  const key = `rate_limit:${ip}`;
  const existing = rateLimitStore.get(key);
  
  if (!existing || now >= existing.resetTime) {
    // Reset the window
    const resetTime = now + RATE_LIMIT_WINDOW;
    rateLimitStore.set(key, { count: 1, resetTime });
    return { allowed: true, remaining: RATE_LIMIT_MAX_REQUESTS - 1, resetTime };
  }
  
  if (existing.count >= RATE_LIMIT_MAX_REQUESTS) {
    return { allowed: false, remaining: 0, resetTime: existing.resetTime };
  }
  
  // Increment count
  existing.count++;
  rateLimitStore.set(key, existing);
  
  return { 
    allowed: true, 
    remaining: RATE_LIMIT_MAX_REQUESTS - existing.count, 
    resetTime: existing.resetTime 
  };
}

function validateRequestBody(body: any, requestId?: string): { valid: boolean; error?: ClaudeError; data?: GenerateRequest } {
  try {
    if (!body || typeof body !== 'object') {
      throw ErrorFactory.createValidationError('Request body must be a JSON object', 'body', body, requestId);
    }
    
    const { userDescription, stockData, mode, securityConfig, forecastDays, dashboardParams, generateOnly, userCode } = body;
    
    // Validate userDescription (not required if executing userCode)
    if (!userCode && (!userDescription || typeof userDescription !== 'string')) {
      throw ErrorFactory.createValidationError('userDescription is required and must be a string', 'userDescription', userDescription, requestId);
    }
    
    if (userDescription && userDescription.length > MAX_DESCRIPTION_LENGTH) {
      throw ErrorFactory.createValidationError(`userDescription must be ${MAX_DESCRIPTION_LENGTH} characters or less`, 'userDescription', userDescription, requestId);
    }
    
    // Validate mode
    if (!mode || (mode !== 'forecast' && mode !== 'backtest')) {
      throw ErrorFactory.createValidationError('mode is required and must be either "forecast" or "backtest"', 'mode', mode, requestId);
    }
    
    // Validate forecastDays for forecast mode
    if (mode === 'forecast' && forecastDays !== undefined) {
      if (typeof forecastDays !== 'number' || !Number.isInteger(forecastDays) || forecastDays < 1 || forecastDays > 365) {
        throw ErrorFactory.createValidationError('forecastDays must be an integer between 1 and 365', 'forecastDays', forecastDays, requestId);
      }
    }
    
    // Validate stockData
    if (!Array.isArray(stockData)) {
      throw ErrorFactory.createValidationError('stockData must be an array', 'stockData', stockData, requestId);
    }
    
    if (stockData.length === 0) {
      throw ErrorFactory.createValidationError('stockData array cannot be empty', 'stockData', stockData, requestId);
    }
    
    if (stockData.length > MAX_STOCK_DATA_COUNT) {
      throw ErrorFactory.createValidationError(`stockData array cannot exceed ${MAX_STOCK_DATA_COUNT} items`, 'stockData', stockData, requestId);
    }
    
    // Validate each stock data item
    for (let i = 0; i < stockData.length; i++) {
      const stock = stockData[i];
      if (!stock || typeof stock !== 'object') {
        throw ErrorFactory.createValidationError(`stockData[${i}] must be an object`, `stockData[${i}]`, stock, requestId);
      }
      
      if (!stock.symbol || typeof stock.symbol !== 'string') {
        throw ErrorFactory.createValidationError(`stockData[${i}].symbol is required and must be a string`, `stockData[${i}].symbol`, stock.symbol, requestId);
      }
      
      if (typeof stock.price !== 'number' || !isFinite(stock.price) || stock.price <= 0) {
        throw ErrorFactory.createValidationError(`stockData[${i}].price must be a positive number`, `stockData[${i}].price`, stock.price, requestId);
      }
    }
    
    // Validate securityConfig if provided
    if (securityConfig && typeof securityConfig !== 'object') {
      throw ErrorFactory.createValidationError('securityConfig must be an object if provided', 'securityConfig', securityConfig, requestId);
    }
    
    return { valid: true, data: { userDescription, stockData, mode, securityConfig, forecastDays, dashboardParams, generateOnly, userCode } };
  } catch (error: any) {
    if (error instanceof ClaudeError) {
      return { valid: false, error };
    }
    return { valid: false, error: ErrorFactory.createValidationError(error?.message || 'Validation failed', undefined, undefined, requestId) };
  }
}

function createErrorResponse(error: string | ClaudeError, statusCode?: number): NextResponse<ErrorResponse> {
  let errorMessage: string;
  let finalStatusCode: number;
  let userMessage: string;
  let suggestions: string[];
  
  if (error instanceof ClaudeError) {
    errorMessage = error.message;
    finalStatusCode = statusCode || (error instanceof ClaudeApiError ? error.statusCode || 500 : 500);
    userMessage = error.getUserMessage();
    suggestions = error.getSuggestions();
    
    // Log the full error details
    console.error('Claude API Error:', error.toLogObject());
  } else {
    errorMessage = error;
    finalStatusCode = statusCode || 400;
    userMessage = error;
    suggestions = [];
  }
  
  return NextResponse.json(
    { 
      success: false, 
      error: userMessage || errorMessage, 
      statusCode: finalStatusCode,
      suggestions
    },
    { 
      status: finalStatusCode,
      headers: {
        'Content-Type': 'application/json',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
      }
    }
  );
}

function createSuccessResponse(result: GenerationResult, rateLimitInfo?: { remaining: number; resetTime: number }): NextResponse<GenerateResponse> {
  return NextResponse.json(
    {
      success: true,
      result,
      ...(rateLimitInfo && { rateLimitInfo }),
    },
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        ...(rateLimitInfo && {
          'X-RateLimit-Remaining': rateLimitInfo.remaining.toString(),
          'X-RateLimit-Reset': Math.ceil(rateLimitInfo.resetTime / 1000).toString(),
        }),
      }
    }
  );
}

// Main API Route Handler
export async function POST(request: NextRequest): Promise<NextResponse> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  try {
    // 1. Validate API Key
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      console.error('ANTHROPIC_API_KEY not configured');
      const error = new ClaudeApiError('Service temporarily unavailable - API key not configured', 503, null, requestId);
      return createErrorResponse(error);
    }
    
    // 2. Rate Limiting
    const clientIP = getClientIP(request);
    const rateLimitCheck = checkRateLimit(clientIP);
    
    if (!rateLimitCheck.allowed) {
      const error = new RateLimitError(
        'Rate limit exceeded. Please try again later.',
        0,
        rateLimitCheck.resetTime,
        requestId
      );
      const response = createErrorResponse(error, 429);
      response.headers.set('X-RateLimit-Remaining', '0');
      response.headers.set('X-RateLimit-Reset', Math.ceil(rateLimitCheck.resetTime / 1000).toString());
      response.headers.set('Retry-After', Math.ceil((rateLimitCheck.resetTime - Date.now()) / 1000).toString());
      return response;
    }
    
    // 3. Parse and Validate Request Body
    let requestBody: any;
    try {
      requestBody = await request.json();
    } catch (parseError) {
      const error = ErrorFactory.createValidationError('Invalid JSON in request body', 'body', undefined, requestId);
      return createErrorResponse(error);
    }
    
    const validation = validateRequestBody(requestBody, requestId);
    if (!validation.valid) {
      return createErrorResponse(validation.error!);
    }
    
    const { userDescription, stockData, mode, securityConfig, forecastDays, dashboardParams, generateOnly, userCode } = validation.data!;
    
    // 4. Security Validation
    const secConfig = securityConfig ? createSecurityConfig(securityConfig) : createSecurityConfig({});
    const securityValidation = validateSecurity(userDescription, undefined, secConfig);
    
    if (!securityValidation.overallValid) {
      const error = ErrorFactory.createSecurityError(
        `Security validation failed: ${securityValidation.promptValidation.blockedReason}`,
        securityValidation.promptValidation.blockedReason || 'Security violation detected',
        'HIGH',
        requestId
      );
      return createErrorResponse(error, 400);
    }
    
    // 5. Initialize Claude API
    let anthropic: Anthropic;
    try {
      anthropic = new Anthropic({
        apiKey: apiKey,
      });
    } catch (initError: any) {
      console.error('Failed to initialize Anthropic client:', initError);
      const error = new ClaudeApiError('Service initialization failed', 500, initError, requestId);
      return createErrorResponse(error);
    }
    
    // 6. Create Claude API Call Function
    const claudeApiCall = async (prompt: string): Promise<string> => {
      try {
        const message = await anthropic.messages.create({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4000,
          temperature: 0.3,
          messages: [
            {
              role: 'user',
              content: prompt,
            },
          ],
        });
        
        if (message.content[0].type === 'text') {
          return message.content[0].text;
        } else {
          throw new Error('Unexpected response format from Claude API');
        }
      } catch (claudeError: any) {
        console.error('Claude API error:', claudeError);
        
        // Handle specific Claude API errors with enhanced error types
        if (claudeError.status === 429) {
          throw new RateLimitError('Claude API rate limit exceeded. Please try again later.', 0, Date.now() + 60000, requestId);
        } else if (claudeError.status === 401) {
          throw new ClaudeApiError('Claude API authentication failed', 401, claudeError, requestId);
        } else if (claudeError.status >= 500) {
          throw new ClaudeApiError('Claude API service unavailable', claudeError.status, claudeError, requestId);
        } else {
          throw new ClaudeApiError(claudeError.message || 'Unknown Claude API error', claudeError.status, claudeError, requestId);
        }
      }
    };
    
    // 7. Handle different request types
    const processRequest = async (): Promise<GenerationResult | { success: boolean; code?: string; error?: string }> => {
      return new Promise(async (resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new TimeoutError('Request timeout', REQUEST_TIMEOUT, requestId));
        }, REQUEST_TIMEOUT);
        
        try {
          let result: GenerationResult | { success: boolean; code?: string; error?: string };
          
          if (userCode) {
            // Execute user-provided code
            result = await executeUserCode(
              userCode,
              stockData,
              mode,
              forecastDays || 30,
              dashboardParams,
              secConfig
            );
          } else if (generateOnly) {
            // Generate code only, don't execute
            result = await generateCodeOnly(
              userDescription,
              stockData,
              mode,
              claudeApiCall,
              secConfig,
              forecastDays || 30,
              dashboardParams
            );
          } else {
            // Original flow: generate and execute
            result = await generatePortfolioWeights(
              userDescription,
              stockData,
              mode,
              claudeApiCall,
              secConfig,
              forecastDays || 30,
              dashboardParams
            );
          }
          
          clearTimeout(timeout);
          resolve(result);
        } catch (error) {
          clearTimeout(timeout);
          reject(error);
        }
      });
    };
    
    // 8. Execute Request
    let result: GenerationResult | { success: boolean; code?: string; error?: string };
    try {
      result = await processRequest();
    } catch (generationError: any) {
      console.error('Portfolio generation error:', generationError);
      
      // Handle specific error types with fallback strategies
      if (generationError instanceof TimeoutError) {
        return createErrorResponse(generationError, 408);
      }
      
      if (generationError instanceof ClaudeError) {
        // For retryable errors, provide fallback instead of failing
        if (!generationError.isRetryable()) {
          return createErrorResponse(generationError);
        }
      }
      
      // For other errors, provide a fallback result
      if (mode === 'backtest') {
        const fallbackWeights = stockData.map(() => 1.0 / stockData.length);
        result = {
          success: true,
          type: mode,
          weights: fallbackWeights,
          error: `Generation failed: ${generationError.message}. Using equal weights fallback.`,
          fallbackUsed: true,
          executionTime: 0,
        };
      } else {
        // Forecast fallback
        const predictions = [];
        const startDate = new Date();
        const days = forecastDays || 30;
        
        for (let i = 1; i <= days; i++) {
          const futureDate = new Date(startDate);
          futureDate.setDate(startDate.getDate() + i);
          
          predictions.push({
            date: futureDate.toISOString().split('T')[0],
            price: stockData[0].price * (1 + (Math.random() - 0.5) * 0.01),
            confidence: 0.5
          });
        }
        
        result = {
          success: true,
          type: mode,
          predictions: predictions,
          error: `Generation failed: ${generationError.message}. Using simple trend fallback.`,
          fallbackUsed: true,
          executionTime: 0,
        };
      }
    }
    
    // 9. Return Success Response
    if (generateOnly && 'code' in result) {
      // For code-only generation, return special response format
      return NextResponse.json({
        success: result.success,
        code: result.code,
        error: result.error,
        rateLimitInfo: {
          remaining: rateLimitCheck.remaining,
          resetTime: rateLimitCheck.resetTime,
        }
      });
    } else {
      // Normal generation result
      return createSuccessResponse(result as GenerationResult, {
        remaining: rateLimitCheck.remaining,
        resetTime: rateLimitCheck.resetTime,
      });
    }
    
  } catch (error: any) {
    // 10. Global Error Handler
    console.error('Unexpected error in Claude API route:', error);
    
    if (error instanceof ClaudeError) {
      return createErrorResponse(error);
    }
    
    // Handle unknown errors
    const unknownError = ErrorFactory.createFromFetchError(error, requestId);
    return createErrorResponse(unknownError);
  }
}

// Handle unsupported HTTP methods
export async function GET(): Promise<NextResponse> {
  return createErrorResponse('Method not allowed. Use POST.', 405);
}

export async function PUT(): Promise<NextResponse> {
  return createErrorResponse('Method not allowed. Use POST.', 405);
}

export async function DELETE(): Promise<NextResponse> {
  return createErrorResponse('Method not allowed. Use POST.', 405);
}

export async function PATCH(): Promise<NextResponse> {
  return createErrorResponse('Method not allowed. Use POST.', 405);
}