/**
 * Enhanced error handling system for Claude integration.
 * Provides typed errors with severity, retry logic, and user-friendly messages.
 */

import { API_REQUEST_TIMEOUT } from './constants';

// Error severity levels
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Error categories
export enum ErrorCategory {
  API = 'api',
  NETWORK = 'network',
  VALIDATION = 'validation',
  EXECUTION = 'execution',
  AUTHENTICATION = 'authentication',
  RATE_LIMIT = 'rate_limit',
  SECURITY = 'security',
  TIMEOUT = 'timeout',
  PARSE = 'parse',
  UNKNOWN = 'unknown'
}

// Base Claude Error class
export abstract class ClaudeError extends Error {
  public readonly timestamp: number;
  public readonly requestId?: string;

  constructor(
    message: string,
    public readonly code: string,
    public readonly category: ErrorCategory,
    public readonly severity: ErrorSeverity,
    public readonly retryable: boolean = false,
    public readonly userMessage?: string,
    public readonly suggestions?: string[],
    requestId?: string
  ) {
    super(message);
    this.name = this.constructor.name;
    this.timestamp = Date.now();
    this.requestId = requestId;

    // Ensure proper prototype chain for instanceof checks
    Object.setPrototypeOf(this, new.target.prototype);
  }

  // Get user-friendly error message
  getUserMessage(): string {
    return this.userMessage || this.message;
  }

  // Get actionable suggestions
  getSuggestions(): string[] {
    return this.suggestions || [];
  }

  // Check if error should be retried
  isRetryable(): boolean {
    return this.retryable;
  }

  // Get error details for logging
  toLogObject() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      category: this.category,
      severity: this.severity,
      retryable: this.retryable,
      timestamp: this.timestamp,
      requestId: this.requestId,
      stack: this.stack
    };
  }
}

// API-related errors
export class ClaudeApiError extends ClaudeError {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly responseBody?: any,
    requestId?: string
  ) {
    const code = `API_ERROR_${statusCode || 'UNKNOWN'}`;
    const severity = statusCode && statusCode >= 500 ? ErrorSeverity.HIGH : ErrorSeverity.MEDIUM;
    const retryable = statusCode ? [429, 500, 502, 503, 504].includes(statusCode) : false;

    let userMessage: string;
    let suggestions: string[];

    switch (statusCode) {
      case 401:
        userMessage = "Authentication failed with Claude API";
        suggestions = ["Check your API key configuration", "Contact administrator"];
        break;
      case 429:
        userMessage = "Too many requests. Please wait before trying again";
        suggestions = ["Wait a few minutes before retrying", "Try simplifying your request"];
        break;
      case 500:
      case 502:
      case 503:
      case 504:
        userMessage = "Claude service is temporarily unavailable";
        suggestions = ["Try again in a few minutes", "Check Claude service status"];
        break;
      default:
        userMessage = "Failed to communicate with Claude API";
        suggestions = ["Check your internet connection", "Try again later"];
    }

    super(message, code, ErrorCategory.API, severity, retryable, userMessage, suggestions, requestId);
  }
}

// Network-related errors
export class NetworkError extends ClaudeError {
  constructor(
    message: string,
    public readonly originalError?: any,
    requestId?: string
  ) {
    super(
      message,
      'NETWORK_ERROR',
      ErrorCategory.NETWORK,
      ErrorSeverity.MEDIUM,
      true,
      "Network connection failed",
      [
        "Check your internet connection",
        "Try again in a few moments",
        "Contact support if problem persists"
      ],
      requestId
    );
  }
}

// Validation errors
export class ValidationError extends ClaudeError {
  constructor(
    message: string,
    public readonly field?: string,
    public readonly value?: any,
    requestId?: string
  ) {
    super(
      message,
      'VALIDATION_ERROR',
      ErrorCategory.VALIDATION,
      ErrorSeverity.LOW,
      false,
      "Invalid input provided",
      [
        "Check your input and try again",
        "Ensure all required fields are filled",
        "Follow the format guidelines"
      ],
      requestId
    );
  }
}

// Code execution errors
export class ExecutionError extends ClaudeError {
  constructor(
    message: string,
    public readonly generatedCode?: string,
    public readonly executionContext?: string,
    requestId?: string
  ) {
    super(
      message,
      'EXECUTION_ERROR',
      ErrorCategory.EXECUTION,
      ErrorSeverity.MEDIUM,
      true,
      "Generated strategy code failed to execute",
      [
        "Try simplifying your strategy description",
        "Use more common investment terms",
        "Generate a new strategy"
      ],
      requestId
    );
  }
}

// Timeout errors
export class TimeoutError extends ClaudeError {
  constructor(
    message: string,
    public readonly timeoutMs: number,
    requestId?: string
  ) {
    super(
      message,
      'TIMEOUT_ERROR',
      ErrorCategory.TIMEOUT,
      ErrorSeverity.MEDIUM,
      true,
      "Request timed out",
      [
        "Try simplifying your strategy description",
        "Check your internet connection",
        "Try again with a shorter request"
      ],
      requestId
    );
  }
}

// Rate limiting errors
export class RateLimitError extends ClaudeError {
  constructor(
    message: string,
    public readonly remaining: number = 0,
    public readonly resetTime: number = Date.now() + 60000,
    requestId?: string
  ) {
    const waitMinutes = Math.ceil((resetTime - Date.now()) / 60000);

    super(
      message,
      'RATE_LIMIT_ERROR',
      ErrorCategory.RATE_LIMIT,
      ErrorSeverity.MEDIUM,
      true,
      "Too many requests made recently",
      [
        `Wait ${waitMinutes} minute${waitMinutes !== 1 ? 's' : ''} before trying again`,
        "Consider reducing request frequency",
        "Try a simpler strategy description"
      ],
      requestId
    );
  }
}

// Parse errors
export class ParseError extends ClaudeError {
  constructor(
    message: string,
    public readonly rawResponse?: string,
    requestId?: string
  ) {
    super(
      message,
      'PARSE_ERROR',
      ErrorCategory.PARSE,
      ErrorSeverity.MEDIUM,
      true,
      "Failed to process Claude's response",
      [
        "Try generating the strategy again",
        "Simplify your strategy description",
        "Contact support if problem persists"
      ],
      requestId
    );
  }
}

// Security errors
export class SecurityError extends ClaudeError {
  constructor(
    message: string,
    public readonly blockedReason: string,
    public readonly riskLevel?: string,
    requestId?: string
  ) {
    super(
      message,
      'SECURITY_ERROR',
      ErrorCategory.SECURITY,
      ErrorSeverity.HIGH,
      false,
      "Strategy blocked for security reasons",
      [
        "Revise your strategy to focus on legitimate investment approaches",
        "Avoid requesting potentially harmful operations",
        "Use standard investment terminology"
      ],
      requestId
    );
  }
}

// Error factory for creating appropriate error types
export class ErrorFactory {
  static createFromApiResponse(
    response: Response,
    responseBody?: any,
    requestId?: string
  ): ClaudeError {
    const { status, statusText } = response;
    const message = responseBody?.error || statusText || 'API request failed';

    return new ClaudeApiError(message, status, responseBody, requestId);
  }

  static createFromFetchError(
    error: any,
    requestId?: string
  ): ClaudeError {
    if (error.name === 'AbortError' || error.code === 'TIMEOUT') {
      return new TimeoutError(
        error.message || 'Request timed out',
        API_REQUEST_TIMEOUT,
        requestId
      );
    }

    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return new NetworkError(
        'Network request failed',
        error,
        requestId
      );
    }

    return new NetworkError(
      error.message || 'Unknown error occurred',
      error,
      requestId
    );
  }

  static createValidationError(
    message: string,
    field?: string,
    value?: any,
    requestId?: string
  ): ValidationError {
    return new ValidationError(message, field, value, requestId);
  }

  static createExecutionError(
    message: string,
    code?: string,
    context?: string,
    requestId?: string
  ): ExecutionError {
    return new ExecutionError(message, code, context, requestId);
  }

  static createSecurityError(
    message: string,
    blockedReason: string,
    riskLevel?: string,
    requestId?: string
  ): SecurityError {
    return new SecurityError(message, blockedReason, riskLevel, requestId);
  }
}

// Error aggregator for handling multiple errors
export class ErrorAggregator {
  private errors: ClaudeError[] = [];

  add(error: ClaudeError): void {
    this.errors.push(error);
  }

  hasErrors(): boolean {
    return this.errors.length > 0;
  }

  getErrors(): ClaudeError[] {
    return [...this.errors];
  }

  getHighestSeverity(): ErrorSeverity {
    if (this.errors.length === 0) return ErrorSeverity.LOW;

    const severityOrder = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL];
    return this.errors.reduce((highest: ErrorSeverity, error: ClaudeError) => {
      const currentIndex = severityOrder.indexOf(error.severity);
      const highestIndex = severityOrder.indexOf(highest);
      return currentIndex > highestIndex ? error.severity : highest;
    }, ErrorSeverity.LOW);
  }

  getMostCriticalError(): ClaudeError | null {
    if (this.errors.length === 0) return null;

    const severityOrder = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL];
    return this.errors.reduce((mostCritical, error) => {
      const currentIndex = severityOrder.indexOf(error.severity);
      const criticalIndex = severityOrder.indexOf(mostCritical.severity);
      return currentIndex > criticalIndex ? error : mostCritical;
    });
  }

  clear(): void {
    this.errors = [];
  }
}

// Error utilities
export class ErrorUtils {
  static isRetryableError(error: any): boolean {
    return error instanceof ClaudeError && error.isRetryable();
  }

  static getRetryDelay(attempt: number, baseDelay: number = 1000): number {
    // Exponential backoff with jitter
    const delay = baseDelay * Math.pow(2, attempt - 1);
    const jitter = Math.random() * 0.1 * delay;
    return Math.min(delay + jitter, API_REQUEST_TIMEOUT); // Max equals API timeout
  }

  static shouldShowToUser(error: any): boolean {
    if (!(error instanceof ClaudeError)) return true;
    return error.severity !== ErrorSeverity.LOW;
  }

  static formatUserMessage(error: any): string {
    if (error instanceof ClaudeError) {
      return error.getUserMessage();
    }
    return error.message || 'An unexpected error occurred';
  }

  static formatSuggestions(error: any): string[] {
    if (error instanceof ClaudeError) {
      return error.getSuggestions();
    }
    return ['Try again', 'Contact support if problem persists'];
  }
}
