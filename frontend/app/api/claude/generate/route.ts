import { NextRequest, NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import {
  generateCodeOnlyServer as generateCodeOnly,
  createSecurityConfig,
  ClaudeError,
  ClaudeApiError,
  TimeoutError,
  RateLimitError,
  ErrorFactory,
} from '../../../../lib/claude';
import { requireAuth } from '@/lib/auth/server';
import { createServerApiKeyService } from '@/lib/services/api-keys';

// Response Type Interfaces
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

// In-memory rate limiting store
// Note: Resets on server restart and doesn't sync across multiple server instances.
// For production at scale, consider Redis or a database-backed solution.
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

// Main API Route Handler
export async function POST(request: NextRequest): Promise<NextResponse> {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  try {
    // 1. Require Authentication
    let user;
    try {
      user = await requireAuth();
    } catch {
      return NextResponse.json(
        { success: false, error: 'Please log in to use AI features.', statusCode: 401 },
        { status: 401 }
      );
    }

    // 2. Get User's API Key (no fallback to system key)
    const keyService = createServerApiKeyService();
    const apiKey = await keyService.getKey(user.id, 'anthropic');

    if (!apiKey) {
      return NextResponse.json(
        {
          success: false,
          error: 'Please add your Anthropic API key in Settings to use AI features.',
          statusCode: 400,
          suggestions: ['Go to Settings and add your Anthropic API key']
        },
        { status: 400 }
      );
    }

    // 3. Rate Limiting
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
    
    // 4. Parse Request Body (client.ts already validates)
    let requestBody: any;
    try {
      requestBody = await request.json();
    } catch (parseError) {
      const error = ErrorFactory.createValidationError('Invalid JSON in request body', 'body', undefined, requestId);
      return createErrorResponse(error);
    }

    const { userDescription, stockData, mode, securityConfig, forecastDays } = requestBody;

    // 5. Create security config (validation happens in generateCodeOnly)
    const secConfig = securityConfig ? createSecurityConfig(securityConfig) : createSecurityConfig({});

    // 6. Initialize Claude API
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
    
    // 7. Create Claude API Call Function
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
    
    // 8. Generate code with timeout
    const processRequest = async (): Promise<{ success: boolean; code?: string; error?: string }> => {
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new TimeoutError('Request timeout', REQUEST_TIMEOUT, requestId)), REQUEST_TIMEOUT);
      });

      return Promise.race([
        generateCodeOnly(userDescription, stockData, mode, claudeApiCall, secConfig, forecastDays || 30),
        timeoutPromise
      ]);
    };
    
    // 9. Execute Request
    let result: { success: boolean; code?: string; error?: string };
    try {
      result = await processRequest();
    } catch (generationError: any) {
      console.error('Code generation error:', generationError);

      if (generationError instanceof TimeoutError) {
        return createErrorResponse(generationError, 408);
      }

      if (generationError instanceof ClaudeError) {
        return createErrorResponse(generationError);
      }

      return createErrorResponse(generationError.message || 'Generation failed', 500);
    }

    // 10. Return Success Response
    return NextResponse.json({
      success: result.success,
      code: result.code,
      error: result.error,
      rateLimitInfo: {
        remaining: rateLimitCheck.remaining,
        resetTime: rateLimitCheck.resetTime,
      }
    });
    
  } catch (error: any) {
    // 11. Global Error Handler
    console.error('Unexpected error in Claude API route:', error);
    
    if (error instanceof ClaudeError) {
      return createErrorResponse(error);
    }
    
    // Handle unknown errors
    const unknownError = ErrorFactory.createFromFetchError(error, requestId);
    return createErrorResponse(unknownError);
  }
}

