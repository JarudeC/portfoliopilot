"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Link from "next/link";
import { generateCodeOnly, executeUserCode, type GenerationResult, type StockData } from "../../lib/claude";
import CodeEditor from './CodeEditor';

/**
 * Extract user-friendly error message from any error.
 * API key validation happens before this point, so we only handle
 * network, rate limit, and execution errors here.
 */
function getErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    if (msg.includes('rate limit') || msg.includes('too many')) {
      return "Too many requests. Please wait a moment before trying again.";
    }
    if (msg.includes('network') || msg.includes('timeout') || msg.includes('fetch')) {
      return "Network issue. Please check your connection and try again.";
    }
    if (msg.includes('duplicate') || msg.includes('already being processed')) {
      return "A similar request is already being processed.";
    }
    return err.message;
  }
  return "An unexpected error occurred. Please try again.";
}

// Popup component props
interface ClaudePopupProps {
  isOpen: boolean;
  onClose: () => void;
  stockData?: StockData[];
  onStrategyGenerated?: (result: GenerationResult) => void;
  onError?: (error: Error) => void;
  mode: 'forecast' | 'backtest';
  // Dashboard parameters
  dashboardParams?: {
    // Forecast params
    historyDays?: number;
    forecastDays?: number;
    // Backtest params  
    backtestDays?: number;
    lookbackDays?: number;
    evaluationWindow?: number;
    transactionCost?: number;
    algorithm?: string;
  };
}

// Character limits and validation
const MAX_DESCRIPTION_LENGTH = 500;
const MIN_DESCRIPTION_LENGTH = 10;

// Predefined examples - backtesting strategies using available data: symbol, price, lookbackPrices, lookbackDates
const BACKTEST_EXAMPLES = [
  "Momentum strategy. Calculate the total return over the lookback period for each stock using lookbackPrices. Assign higher weights to stocks with positive momentum, zero weight to stocks with negative returns.",
  "Low volatility strategy. Calculate the standard deviation of daily returns from lookbackPrices for each stock. Assign higher weights to stocks with lower volatility - inverse of volatility as weight.",
  "Recent trend strategy. Calculate the slope of prices over the last 10 days using linear regression on lookbackPrices. Weight stocks with positive slopes higher, zero weight for negative slopes.",
  "Mean reversion strategy. Compare current price to the average of lookbackPrices. Assign higher weights to stocks trading below their average (oversold), lower weights to stocks above average.",
  "Risk-adjusted momentum. Calculate both momentum (total return) and volatility from lookbackPrices. Weight = momentum / volatility for positive momentum stocks, zero for negative momentum."
];

const FORECAST_EXAMPLES = [
  "Exponential smoothing with trend. Calculate exponentially weighted average of last 20 prices with alpha=0.3, detect linear trend from last 10 days, project forward with 90% trend persistence and decreasing confidence.",
  "ARIMA-style forecast. Calculate 5-day moving average as baseline, add autoregressive component using correlation with 1-day and 2-day lags, smooth daily changes to maximum 2% up or down.",
  "Bollinger band mean reversion. Calculate 20-day moving average and 2-sigma bands, predict gradual convergence toward moving average over forecast period with 70% reversion strength.",
  "Momentum persistence with decay. Calculate 10-day price momentum, project forward with exponentially decaying strength (starts at 80%, decays by 5% each day), bounded by ±1.5% daily moves.",
  "Volatility-adjusted random walk. Use last price as starting point, calculate 30-day historical volatility, generate daily moves from normal distribution with mean=0 and std=volatility/3 for stability."
];

export default function ClaudePopup({ 
  isOpen,
  onClose,
  stockData, 
  onStrategyGenerated, 
  onError, 
  mode,
  dashboardParams = {}
}: ClaudePopupProps) {
  // Dynamic content based on mode
  const examples = mode === 'forecast' ? FORECAST_EXAMPLES : BACKTEST_EXAMPLES;
  const title = mode === 'forecast' ? 'Custom AI Forecast Strategy' : 'Custom AI Backtest Strategy';
  
  const description = mode === 'forecast' 
    ? `Generate price predictions for ${dashboardParams.forecastDays || 14} days using ${dashboardParams.historyDays || 180} days of history`
    : `Generate portfolio weights using ${dashboardParams.backtestDays || 365} days of backtest data`;
    
  const getPlaceholder = () => {
    if (mode === 'forecast') {
      return `Describe your forecasting algorithm...

Examples:
• Exponential smoothing with trend. Calculate exponentially weighted average of last 20 prices with alpha=0.3, detect linear trend, project forward with 90% trend persistence.
• ARIMA-style forecast. Calculate 5-day moving average as baseline, add autoregressive component using 1-day and 2-day lags, smooth daily changes to maximum 2%.
• Bollinger band mean reversion. Calculate 20-day moving average and 2-sigma bands, predict gradual convergence toward moving average with 70% reversion strength.

The system will use your selected parameters:
• History Days: ${dashboardParams.historyDays || 180}
• Forecast Days: ${dashboardParams.forecastDays || 14}
• Base Algorithm: ${dashboardParams.algorithm || 'LSTM'}

Press Ctrl+Enter to generate, Esc to close`;
    } else {
      return `Describe your investment strategy...

Available data per stock: symbol, price, lookbackPrices (array), lookbackDates (array)

Examples:
• Momentum strategy. Calculate total return over lookback period using lookbackPrices. Higher weights for positive momentum.
• Low volatility. Calculate standard deviation of daily returns from lookbackPrices. Higher weights for lower volatility.
• Mean reversion. Compare current price to average of lookbackPrices. Higher weights for stocks trading below average.

The system will use your selected parameters:
• Backtest Days: ${dashboardParams.backtestDays || 365}
• Lookback Days: ${dashboardParams.lookbackDays || 30}
• Evaluation Window: ${dashboardParams.evaluationWindow || 5}
• Transaction Cost: ${(dashboardParams.transactionCost || 0.002) * 100}%
• Base Algorithm: ${dashboardParams.algorithm || 'Naive Markowitz'}

Press Ctrl+Enter to generate, Esc to close`;
    }
  };

  // State Management
  const [userDescription, setUserDescription] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<GenerationResult | null>(null);

  // Code review states
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [showCodeReview, setShowCodeReview] = useState<boolean>(false);
  const [executingCode, setExecutingCode] = useState<boolean>(false);

  // API key status
  const [hasApiKey, setHasApiKey] = useState<boolean | null>(null);
  const [checkingApiKey, setCheckingApiKey] = useState<boolean>(true);

  // Refs for DOM elements
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Character count calculations
  const charactersUsed = userDescription.length;
  const charactersRemaining = MAX_DESCRIPTION_LENGTH - charactersUsed;
  const isDescriptionValid = charactersUsed >= MIN_DESCRIPTION_LENGTH && charactersUsed <= MAX_DESCRIPTION_LENGTH;
  const hasStocksSelected = stockData && stockData.length > 0;

  // Input validation
  const validateInput = useCallback((input: string): string | null => {
    if (input.trim().length < MIN_DESCRIPTION_LENGTH) {
      return `Description must be at least ${MIN_DESCRIPTION_LENGTH} characters`;
    }
    if (input.length > MAX_DESCRIPTION_LENGTH) {
      return `Description must be ${MAX_DESCRIPTION_LENGTH} characters or less`;
    }
    if (!/[a-zA-Z]/.test(input)) {
      return "Description must contain at least some letters";
    }
    return null;
  }, []);

  // Handle description change with validation
  const handleDescriptionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setUserDescription(value);
    
    // Real-time validation
    const validation = validateInput(value);
    setValidationError(validation);
    
    // Clear previous errors when user starts typing
    if (error) setError(null);
  }, [validateInput, error]);

  // Generate code function - first step: Generate code only for review
  const handleGenerate = useCallback(async () => {
    if (loading || !stockData || stockData.length === 0) return;

    // Final validation
    const validation = validateInput(userDescription);
    if (validation) {
      setValidationError(validation);
      textareaRef.current?.focus();
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setValidationError(null);

      // Call the new generateCodeOnly function to get code without execution
      const result = await generateCodeOnly(
        userDescription.trim(),
        mode,
        stockData, // Use actual user-selected stocks
        undefined, // securityConfig
        mode === 'forecast' ? dashboardParams.forecastDays : undefined
      );
      
      if (result.success && result.code) {
        setGeneratedCode(result.code);
        setShowCodeReview(true);
      } else {
        setError(result.error || "No code was generated. Please try a different description.");
      }
      
    } catch (err) {
      console.error("Code generation error:", err);
      setError(getErrorMessage(err));
      onError?.(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [userDescription, stockData, loading, validateInput, onError, mode, dashboardParams]);

  // Execute approved code function - second step
  const handleCodeApproval = useCallback(async (editedCode: string) => {
    if (executingCode || !stockData || stockData.length === 0) return;

    try {
      setExecutingCode(true);
      setError(null);

      
      const result = await executeUserCode(
        editedCode,
        mode, 
        stockData,
        undefined,
        mode === 'forecast' ? dashboardParams.forecastDays : undefined,
        dashboardParams
      );
      
      // Update state with results
      setLastResult(result);
      setShowCodeReview(false);
      setGeneratedCode(null);

      // Notify parent component
      if (onStrategyGenerated) {
        onStrategyGenerated(result);
      }

      // Show success message briefly
      if (result.fallbackUsed) {
        setError("Execution failed, using fallback strategy. The edited code couldn't be executed safely.");
      }

    } catch (err) {
      console.error("Strategy execution error:", err);
      setError(getErrorMessage(err));
      onError?.(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setExecutingCode(false);
    }
  }, [stockData, executingCode, onStrategyGenerated, onError, mode, dashboardParams]);

  // Handle code rejection
  const handleCodeRejection = useCallback(() => {
    setShowCodeReview(false);
    setGeneratedCode(null);
  }, []);

  // Clear all data
  const handleClear = useCallback(() => {
    setUserDescription("");
    setError(null);
    setValidationError(null);
    setLastResult(null);
    setGeneratedCode(null);
    setShowCodeReview(false);
    textareaRef.current?.focus();
  }, []);

  // Use example strategy
  const handleUseExample = useCallback(() => {
    const randomExample = examples[Math.floor(Math.random() * examples.length)];
    setUserDescription(randomExample);
    setValidationError(null);
    if (error) setError(null);
    textareaRef.current?.focus();
  }, [error, examples]);

  // Handle close popup
  const handleClose = useCallback(() => {
    if (loading || executingCode) return;
    handleClear();
    onClose();
  }, [loading, executingCode, handleClear, onClose]);

  // Handle overlay click
  const handleOverlayClick = useCallback((e: React.MouseEvent) => {
    if (e.target === overlayRef.current) {
      handleClose();
    }
  }, [handleClose]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      handleClose();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleGenerate();
    }
  }, [handleClose, handleGenerate]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [userDescription]);

  // Focus management
  useEffect(() => {
    if (isOpen && textareaRef.current && hasApiKey) {
      // Small delay to ensure the modal is fully rendered
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  }, [isOpen, hasApiKey]);

  // Check for API key on mount (runs once per component mount)
  useEffect(() => {
    setCheckingApiKey(true);
    fetch('/api/settings/api-key?provider=anthropic')
      .then(res => res.json())
      .then(data => {
        setHasApiKey(data.hasKey === true);
      })
      .catch(() => {
        // If we can't check, assume no key (will get proper error on generate)
        setHasApiKey(false);
      })
      .finally(() => {
        setCheckingApiKey(false);
      });
  }, []); // Empty dependency = runs once on mount

  if (!isOpen) return null;

  // Show loading state while checking API key
  if (checkingApiKey) {
    return (
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={handleOverlayClick}
      >
        <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl p-8 max-w-md w-full text-center">
          <div className="w-8 h-8 border-2 border-[#4CC9F0]/30 border-t-[#4CC9F0] rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Checking configuration...</p>
        </div>
      </div>
    );
  }

  // Show setup prompt if no API key is configured
  if (!hasApiKey) {
    return (
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={handleOverlayClick}
      >
        <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl max-w-lg w-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-[#4CC9F0]/20">
            <h2 className="text-xl font-bold text-white">API Key Required</h2>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-white transition-colors p-2"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Icon and Message */}
            <div className="text-center">
              <div className="w-16 h-16 bg-[#4CC9F0]/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-[#4CC9F0]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">
                Set Up Your AI Features
              </h3>
              <p className="text-gray-400 text-sm">
                To use AI-powered strategy generation, you need to add your own Anthropic API key.
                Your key is encrypted and stored securely.
              </p>
            </div>

            {/* Steps */}
            <div className="bg-[#0D1B2A] rounded-lg p-4 space-y-3">
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-[#4CC9F0]/20 text-[#4CC9F0] rounded-full flex items-center justify-center text-sm font-medium">1</span>
                <div>
                  <p className="text-white text-sm font-medium">Get an API key</p>
                  <p className="text-gray-400 text-xs">
                    Visit{' '}
                    <a
                      href="https://console.anthropic.com/settings/keys"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#4CC9F0] hover:underline"
                    >
                      console.anthropic.com
                    </a>
                    {' '}to create your API key
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-[#4CC9F0]/20 text-[#4CC9F0] rounded-full flex items-center justify-center text-sm font-medium">2</span>
                <div>
                  <p className="text-white text-sm font-medium">Add it to Settings</p>
                  <p className="text-gray-400 text-xs">
                    Paste your API key in the Settings page to enable AI features
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-[#4CC9F0]/20 text-[#4CC9F0] rounded-full flex items-center justify-center text-sm font-medium">3</span>
                <div>
                  <p className="text-white text-sm font-medium">Start generating strategies</p>
                  <p className="text-gray-400 text-xs">
                    Come back here to create custom AI-powered strategies
                  </p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Link
                href="/settings"
                className="flex-1 bg-[#4CC9F0] hover:bg-[#4CC9F0]/90 text-black font-medium py-3 px-6 rounded-lg transition-colors text-center"
              >
                Go to Settings
              </Link>
              <button
                onClick={handleClose}
                className="px-6 py-3 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show code editor if we have generated code
  if (showCodeReview && generatedCode) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="max-w-6xl w-full max-h-[90vh] overflow-y-auto">
          <CodeEditor
            code={generatedCode}
            onApprove={handleCodeApproval}
            onReject={handleCodeRejection}
            mode={mode}
            loading={executingCode}
          />
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={overlayRef}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={handleOverlayClick}
      onKeyDown={handleKeyDown}
    >
      <div className="bg-[#14273F] border border-[#4CC9F0]/30 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-[#4CC9F0]/20">
          <div>
            <h2 className="text-xl font-bold text-white">{title}</h2>
            <p className="text-gray-400 text-sm mt-1">{description}</p>
          </div>
          <button
            onClick={handleClose}
            disabled={loading}
            className="text-gray-400 hover:text-white transition-colors p-2 disabled:opacity-50"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Input Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="strategy-description" className="block text-sm font-medium text-gray-300">
                Algorithm Description
              </label>
              <button
                type="button"
                onClick={handleUseExample}
                disabled={loading}
                className="text-xs text-[#4CC9F0] hover:text-[#4CC9F0]/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Use Example
              </button>
            </div>

            <div className="relative">
              <textarea
                ref={textareaRef}
                id="strategy-description"
                value={userDescription}
                onChange={handleDescriptionChange}
                disabled={loading}
                placeholder={getPlaceholder()}
                className="w-full min-h-[100px] max-h-[150px] px-4 py-3 bg-[#0D1B2A] border border-[#4CC9F0]/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors resize-none disabled:opacity-50 disabled:cursor-not-allowed"
                rows={4}
              />
            </div>

            {/* Character Counter - Below textarea */}
            <div className="flex justify-end">
              <span className={`text-xs ${charactersRemaining < 50 ? "text-orange-400" : charactersRemaining < 20 ? "text-red-400" : "text-gray-500"}`}>
                {charactersRemaining} characters remaining
              </span>
            </div>
          </div>

          {/* No Stocks Selected Warning */}
          {!hasStocksSelected && (
            <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-yellow-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-yellow-400">Please select at least one stock before generating a strategy</p>
              </div>
            </div>
          )}

          {/* Validation Error */}
          {validationError && (
            <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
              <p className="text-sm text-orange-400">{validationError}</p>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div>
                  <h4 className="text-sm font-medium text-red-400">Error</h4>
                  <p className="text-sm text-red-300 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Success Display */}
          {lastResult && !error && (
            <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-green-400">
                    {lastResult.type === 'backtest' ? 'Strategy Generated Successfully' : 'Forecast Generated Successfully'}
                  </h4>
                  <p className="text-sm text-green-300 mt-1">
                    {lastResult.type === 'backtest' 
                      ? `Generated portfolio weights in ${lastResult.executionTime}ms`
                      : `Generated ${lastResult.predictions?.length || 0} predictions in ${lastResult.executionTime}ms`
                    }
                    {lastResult.fallbackUsed && ` (using fallback ${lastResult.type === 'backtest' ? 'strategy' : 'method'})`}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center gap-3 pt-2">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={loading || !isDescriptionValid || !!validationError || !hasStocksSelected}
              className="flex-1 bg-[#4CC9F0] hover:bg-[#4CC9F0]/90 disabled:bg-gray-600 disabled:cursor-not-allowed text-black font-medium py-3 px-6 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-[#4CC9F0]"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" opacity="0.25"/>
                    <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                  </svg>
                  Generating Code...
                </span>
              ) : (
                'Generate Code'
              )}
            </button>
            
            <button
              type="button"
              onClick={handleClear}
              disabled={loading}
              className="px-4 py-3 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Clear
            </button>

            <button
              type="button"
              onClick={handleClose}
              disabled={loading || executingCode}
              className="px-4 py-3 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Cancel
            </button>
          </div>

          {/* Keyboard Shortcuts Help */}
          <div className="text-center pt-2">
            <p className="text-xs text-gray-500">
              <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Ctrl+Enter</kbd> to generate •{" "}
              <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Esc</kbd> to close
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}