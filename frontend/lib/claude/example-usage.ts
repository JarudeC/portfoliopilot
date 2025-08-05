// Example usage of the Claude client service
// This demonstrates how React components can use the generateStrategy function

import { generateStrategy, ClaudeClientError, ClientErrorType, type StockData } from './client';

// Example 1: Basic usage
export async function basicExample() {
  try {
    const stockData: StockData[] = [
      { symbol: 'AAPL', price: 150.00, marketCap: 2500000000000 },
      { symbol: 'GOOGL', price: 2800.00, marketCap: 1800000000000 },
      { symbol: 'MSFT', price: 300.00, marketCap: 2200000000000 },
      { symbol: 'TSLA', price: 800.00, marketCap: 800000000000 },
    ];

    const result = await generateStrategy(
      'Create a momentum-based portfolio focusing on tech stocks with high growth potential',
      'backtest',
      stockData
    );

    console.log('Generated portfolio weights:', result.weights || 'N/A (forecast mode)');
    console.log('Generated code:', result.code);
    console.log('Execution time:', result.executionTime, 'ms');

    return result;
  } catch (error) {
    if (error instanceof ClaudeClientError) {
      console.error('Claude client error:', error.type, error.message);
      
      switch (error.type) {
        case ClientErrorType.RATE_LIMIT_ERROR:
          console.log('Rate limited. Retry after:', new Date(error.rateLimitInfo?.resetTime || 0));
          break;
        case ClientErrorType.VALIDATION_ERROR:
          console.log('Fix your input:', error.message);
          break;
        default:
          console.log('Unexpected error type:', error.type);
      }
    } else {
      console.error('Unknown error:', error);
    }
    throw error;
  }
}

// Example 2: React component usage pattern
export function usePortfolioGeneration() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ClaudeClientError | null>(null);
  const [result, setResult] = useState<any>(null);

  const generatePortfolio = async (description: string, stocks: StockData[]) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const portfolioResult = await generateStrategy(description, 'backtest', stocks);
      setResult(portfolioResult);
    } catch (err) {
      if (err instanceof ClaudeClientError) {
        setError(err);
      } else {
        setError(new ClaudeClientError(
          ClientErrorType.UNKNOWN_ERROR,
          'An unexpected error occurred'
        ));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return {
    generatePortfolio,
    isLoading,
    error,
    result,
  };
}

// Example 3: Error handling patterns
export function handleClientErrors(error: unknown) {
  if (error instanceof ClaudeClientError) {
    switch (error.type) {
      case ClientErrorType.RATE_LIMIT_ERROR:
        return {
          userMessage: 'You\'ve made too many requests. Please wait a moment before trying again.',
          canRetry: true,
          retryAfter: error.rateLimitInfo?.resetTime,
        };

      case ClientErrorType.VALIDATION_ERROR:
        return {
          userMessage: `Please check your input: ${error.message}`,
          canRetry: false,
        };

      case ClientErrorType.NETWORK_ERROR:
      case ClientErrorType.TIMEOUT_ERROR:
        return {
          userMessage: 'Network connection issue. Please check your internet and try again.',
          canRetry: true,
        };

      case ClientErrorType.API_ERROR:
        if (error.statusCode && error.statusCode >= 500) {
          return {
            userMessage: 'Our service is temporarily unavailable. Please try again later.',
            canRetry: true,
          };
        }
        return {
          userMessage: 'Unable to process your request. Please try a different strategy description.',
          canRetry: false,
        };

      case ClientErrorType.DUPLICATE_REQUEST:
        return {
          userMessage: 'The same request is already being processed.',
          canRetry: false,
        };

      default:
        return {
          userMessage: 'An unexpected error occurred. Please try again.',
          canRetry: true,
        };
    }
  }

  return {
    userMessage: 'An unexpected error occurred. Please try again.',
    canRetry: true,
  };
}

// Example 4: Batch processing with proper error handling
export async function generateMultipleStrategies(strategies: Array<{ name: string; description: string; stocks: StockData[] }>) {
  const results = [];
  
  for (const strategy of strategies) {
    try {
      console.log(`Generating strategy: ${strategy.name}`);
      
      const result = await generateStrategy(strategy.description, 'backtest', strategy.stocks);
      
      results.push({
        name: strategy.name,
        success: true,
        result,
      });
      
      // Add delay between requests to respect rate limits
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } catch (error) {
      console.error(`Failed to generate strategy ${strategy.name}:`, error);
      
      results.push({
        name: strategy.name,
        success: false,
        error: error instanceof ClaudeClientError ? error : new ClaudeClientError(
          ClientErrorType.UNKNOWN_ERROR,
          'Unknown error occurred'
        ),
      });

      // If we hit a rate limit, wait longer before continuing
      if (error instanceof ClaudeClientError && error.type === ClientErrorType.RATE_LIMIT_ERROR) {
        const waitTime = error.rateLimitInfo?.resetTime ? 
          error.rateLimitInfo.resetTime - Date.now() : 
          60000; // Default 1 minute wait
        
        console.log(`Rate limited. Waiting ${waitTime}ms before continuing...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
  }
  
  return results;
}

// Note: This file contains example TypeScript code that would typically need React imports
// In a real React component, you would add: import { useState } from 'react';
declare function useState<T>(initialState: T): [T, (value: T) => void];