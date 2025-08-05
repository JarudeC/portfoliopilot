"use client";

import React, { Component, ReactNode } from 'react';
import { ErrorFactory, ClaudeError, ErrorUtils } from '../../lib/claude/errors';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorId?: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return {
      hasError: true,
      error,
      errorId
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log the error with context
    console.error('ErrorBoundary caught an error:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      errorId: this.state.errorId
    });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Convert to Claude error for consistent handling
    let claudeError: ClaudeError;
    if (error instanceof ClaudeError) {
      claudeError = error;
    } else {
      claudeError = ErrorFactory.createFromFetchError(error, this.state.errorId);
    }

    // Log the structured error
    console.error('Structured error details:', claudeError.toLogObject());
  }

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="text-red-400 mb-4">
            <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 19c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          
          <h3 className="text-lg font-medium text-red-300 mb-2">
            Something went wrong
          </h3>
          
          <p className="text-sm text-red-200/80 text-center mb-4 max-w-md">
            {this.state.error instanceof ClaudeError 
              ? this.state.error.getUserMessage()
              : "An unexpected error occurred while rendering this component."
            }
          </p>

          {this.state.error instanceof ClaudeError && this.state.error.getSuggestions().length > 0 && (
            <div className="mb-4">
              <p className="text-xs text-red-200/60 mb-2">Suggestions:</p>
              <ul className="text-xs text-red-200/80 space-y-1">
                {this.state.error.getSuggestions().map((suggestion, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-red-400 mt-0.5">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          <div className="flex gap-2">
            <button
              onClick={() => this.setState({ hasError: false, error: undefined, errorId: undefined })}
              className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 text-sm rounded-md transition-colors"
            >
              Try Again
            </button>
            
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-gray-500/20 hover:bg-gray-500/30 text-gray-300 text-sm rounded-md transition-colors"
            >
              Reload Page
            </button>
          </div>

          {process.env.NODE_ENV === 'development' && (
            <details className="mt-4 max-w-full">
              <summary className="text-xs text-red-200/60 cursor-pointer hover:text-red-200/80">
                Error Details (Development)
              </summary>
              <pre className="mt-2 text-xs text-red-200/60 bg-red-500/5 p-2 rounded overflow-auto max-h-32">
                {this.state.error?.stack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// Hook version for functional components
export function useErrorBoundary() {
  const [error, setError] = React.useState<Error | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
  }, []);

  const captureError = React.useCallback((error: Error) => {
    setError(error);
  }, []);

  React.useEffect(() => {
    if (error) {
      throw error;
    }
  }, [error]);

  return { captureError, resetError };
}

// Higher-order component wrapper
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}