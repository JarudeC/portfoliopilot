"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { generateStrategy, generateCodeOnly, executeUserCode, ClaudeClientError, ClientErrorType, type GenerationResult, type StockData } from "../../lib/claude/client";
import CodeEditor from './CodeEditor';

// Popup component props
interface ClaudePopupProps {
  isOpen: boolean;
  onClose: () => void;
  stockData?: StockData[];
  onStrategyGenerated?: (result: GenerationResult) => void;
  onError?: (error: ClaudeClientError | Error) => void;
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

// Predefined examples - sophisticated backtesting strategies (optimized for ~1 year training data)
const BACKTEST_EXAMPLES = [
  "Minimum variance optimization. Calculate the covariance matrix of stock returns over 30 days, find portfolio weights that minimize total portfolio variance subject to weights summing to 1.",
  "Maximum Sharpe ratio strategy. For each stock calculate Sharpe ratio as (return - risk_free_rate) / volatility over 45 days. Allocate weights proportional to positive Sharpe ratios, zero to negative ones.",
  "Mean reversion momentum hybrid. Calculate 10-day returns and 3-day returns for each stock. Assign high weights to stocks with negative 10-day returns but positive 3-day returns, indicating mean reversion.",
  "Volatility-adjusted momentum. Calculate 30-day momentum and 15-day volatility for each stock. Set weights to momentum/volatility ratio, normalized to sum to 1, with minimum 5% and maximum 35% per stock.",
  "Quality factor tilt. Use price-to-book ratios from stockData: assign higher weights to stocks with P/B ratios between 0.5-2.0, lower weights to P/B > 3.0, creating a value-growth balanced portfolio."
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

Examples:
• Minimum variance optimization. Calculate the covariance matrix of stock returns over 30 days, find portfolio weights that minimize total portfolio variance.
• Maximum Sharpe ratio strategy. Calculate Sharpe ratio as (return - risk_free_rate) / volatility over 45 days. Allocate weights proportional to positive Sharpe ratios.
• Volatility-adjusted momentum. Calculate 30-day momentum and 15-day volatility for each stock. Set weights to momentum/volatility ratio, normalized to sum to 1.

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

  // Refs for DOM elements
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Character count calculations
  const charactersUsed = userDescription.length;
  const charactersRemaining = MAX_DESCRIPTION_LENGTH - charactersUsed;
  const isDescriptionValid = charactersUsed >= MIN_DESCRIPTION_LENGTH && charactersUsed <= MAX_DESCRIPTION_LENGTH;

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
        mode === 'forecast' ? dashboardParams.forecastDays : undefined,
        dashboardParams
      );
      
      if (result.success && result.code) {
        setGeneratedCode(result.code);
        setShowCodeReview(true);
      } else {
        setError(result.error || "No code was generated. Please try a different description.");
      }
      
    } catch (err: any) {
      console.error("Code generation error:", err);
      
      let errorMessage = "Failed to generate code. Please try again.";
      
      if (err instanceof ClaudeClientError) {
        switch (err.type) {
          case ClientErrorType.RATE_LIMIT_ERROR:
            errorMessage = "Too many requests. Please wait a moment before trying again.";
            break;
          case ClientErrorType.VALIDATION_ERROR:
            errorMessage = `Input validation failed: ${err.message}`;
            break;
          case ClientErrorType.NETWORK_ERROR:
          case ClientErrorType.TIMEOUT_ERROR:
            errorMessage = "Network issue. Please check your connection and try again.";
            break;
          case ClientErrorType.API_ERROR:
            errorMessage = "Service temporarily unavailable. Please try again in a few moments.";
            break;
          case ClientErrorType.DUPLICATE_REQUEST:
            errorMessage = "A similar request is already being processed.";
            break;
          default:
            errorMessage = err.message || errorMessage;
        }
      }
      
      setError(errorMessage);
      
      if (onError) {
        onError(err);
      }
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

    } catch (err: any) {
      console.error("Strategy execution error:", err);
      
      let errorMessage = "An unexpected error occurred. Please try again.";
      
      if (err instanceof ClaudeClientError) {
        switch (err.type) {
          case ClientErrorType.RATE_LIMIT_ERROR:
            errorMessage = "Too many requests. Please wait a moment before trying again.";
            break;
          case ClientErrorType.VALIDATION_ERROR:
            errorMessage = `Input validation failed: ${err.message}`;
            break;
          case ClientErrorType.NETWORK_ERROR:
          case ClientErrorType.TIMEOUT_ERROR:
            errorMessage = "Network issue. Please check your connection and try again.";
            break;
          case ClientErrorType.API_ERROR:
            errorMessage = "Service temporarily unavailable. Please try again in a few moments.";
            break;
          case ClientErrorType.DUPLICATE_REQUEST:
            errorMessage = "A similar request is already being processed.";
            break;
          default:
            errorMessage = err.message || errorMessage;
        }
      }
      
      setError(errorMessage);
      
      // Notify parent component
      if (onError) {
        onError(err);
      }
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
    if (isOpen && textareaRef.current) {
      // Small delay to ensure the modal is fully rendered
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  }, [isOpen]);


  if (!isOpen) return null;

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
              disabled={loading || !isDescriptionValid || !!validationError}
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