"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { generateStrategy, ClaudeClientError, ClientErrorType, type GenerationResult, type StockData } from "../../lib/claude/client";

// Component Props Interface
interface ClaudeStrategyProps {
  stockData?: StockData[];
  onStrategyGenerated?: (result: GenerationResult) => void;
  onError?: (error: ClaudeClientError | Error) => void;
  className?: string;
  disabled?: boolean;
  defaultMode?: 'forecast' | 'backtest';
}

// Character limits and validation
const MAX_DESCRIPTION_LENGTH = 500;
const MIN_DESCRIPTION_LENGTH = 10;

// Predefined example strategies
const BACKTEST_EXAMPLES = [
  "Create a momentum-based portfolio focusing on high-growth tech stocks",
  "Build a dividend-focused portfolio with low volatility for conservative investors",
  "Design a value investing strategy targeting undervalued stocks with strong fundamentals",
  "Develop a sector rotation strategy based on economic cycles",
  "Construct a risk-parity portfolio with equal risk contribution from each asset"
];

const FORECAST_EXAMPLES = [
  "Predict stock prices using technical analysis with moving averages and RSI indicators",
  "Generate price forecasts based on fundamental analysis and earnings trends",
  "Create predictions using momentum indicators and volume analysis",
  "Forecast prices using mean reversion patterns and support/resistance levels",
  "Predict future values based on seasonal trends and market cycles"
];

export default function ClaudeStrategy({ 
  stockData, 
  onStrategyGenerated, 
  onError, 
  className = "",
  disabled = false,
  defaultMode = 'backtest'
}: ClaudeStrategyProps) {
  // State Management  
  const [description, setDescription] = useState<string>("");
  const [mode, setMode] = useState<'forecast' | 'backtest'>(defaultMode);
  const [forecastDays, setForecastDays] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<GenerationResult | null>(null);

  // Refs for DOM elements
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const generateButtonRef = useRef<HTMLButtonElement>(null);

  // Character count calculations
  const charactersUsed = description.length;
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
    setDescription(value);
    
    // Real-time validation
    const validation = validateInput(value);
    setValidationError(validation);
    
    // Clear previous errors when user starts typing
    if (error) setError(null);
  }, [validateInput, error]);

  // Generate strategy function
  const handleGenerate = useCallback(async () => {
    if (loading || disabled) return;

    // Final validation
    const validation = validateInput(description);
    if (validation) {
      setValidationError(validation);
      textareaRef.current?.focus();
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setValidationError(null);

      // Generate strategy using client service
      const result = await generateStrategy(description.trim(), mode, stockData, undefined, forecastDays);
      
      // Update state with results
      setGeneratedCode(result.code || null);
      setLastResult(result);
      setShowPreview(!!result.code);

      // Notify parent component
      if (onStrategyGenerated) {
        onStrategyGenerated(result);
      }

      // Show success message briefly
      if (result.fallbackUsed) {
        setError("Generated using fallback strategy. The AI-generated approach couldn't be used, but we've provided equal weights.");
      }

    } catch (err: any) {
      console.error("Strategy generation error:", err);
      
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
      setLoading(false);
    }
  }, [description, stockData, loading, disabled, validateInput, onStrategyGenerated, onError]);

  // Clear all data
  const handleClear = useCallback(() => {
    setDescription("");
    setError(null);
    setValidationError(null);
    setGeneratedCode(null);
    setLastResult(null);
    setShowPreview(false);
    textareaRef.current?.focus();
  }, []);

  // Use example strategy
  const handleUseExample = useCallback(() => {
    const examples = mode === 'forecast' ? FORECAST_EXAMPLES : BACKTEST_EXAMPLES;
    const randomExample = examples[Math.floor(Math.random() * examples.length)];
    setDescription(randomExample);
    setValidationError(null);
    if (error) setError(null);
    textareaRef.current?.focus();
  }, [error, mode]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleGenerate();
    }
    if (e.key === "Escape") {
      e.preventDefault();
      handleClear();
    }
  }, [handleGenerate, handleClear]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [description]);

  // Focus management
  useEffect(() => {
    if (!loading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [loading]);

  return (
    <div className={`w-full max-w-4xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-white">AI Strategy Generator</h2>
        <p className="text-gray-400 text-sm">
          {mode === 'backtest' 
            ? 'Describe your investment strategy and let AI generate optimized portfolio weights'
            : 'Describe your forecasting strategy and let AI predict future stock prices'
          }
        </p>
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center">
        <div className="bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-lg p-1 flex">
          <button
            type="button"
            onClick={() => setMode('backtest')}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'backtest'
                ? 'bg-[#4CC9F0] text-black'
                : 'text-gray-300 hover:text-white hover:bg-[#4CC9F0]/10'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            Backtest Strategy
          </button>
          <button
            type="button"
            onClick={() => setMode('forecast')}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'forecast'
                ? 'bg-[#4CC9F0] text-black'
                : 'text-gray-300 hover:text-white hover:bg-[#4CC9F0]/10'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            Price Forecast
          </button>
        </div>
      </div>

      {/* Strategy Input Section */}
      <div className="bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-xl p-6 space-y-4">
        {/* Forecast Days Input (only for forecast mode) */}
        {mode === 'forecast' && (
          <div className="flex items-center gap-4">
            <label htmlFor="forecast-days" className="text-sm font-medium text-gray-300">
              Forecast Days:
            </label>
            <input
              id="forecast-days"
              type="number"
              min="1"
              max="365"
              value={forecastDays}
              onChange={(e) => setForecastDays(Math.max(1, Math.min(365, parseInt(e.target.value) || 30)))}
              disabled={loading}
              className="w-20 px-3 py-1 bg-[#0F1419] border border-[#4CC9F0]/20 rounded text-white text-sm focus:outline-none focus:border-[#4CC9F0] disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <span className="text-xs text-gray-500">days into the future</span>
          </div>
        )}

        <div className="flex items-center justify-between">
          <label htmlFor="strategy-description" className="block text-sm font-medium text-gray-300">
            {mode === 'backtest' ? 'Strategy Description' : 'Forecasting Method'}
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
            value={description}
            onChange={handleDescriptionChange}
            onKeyDown={handleKeyDown}
            disabled={loading || disabled}
            placeholder={mode === 'backtest' 
              ? `Describe your investment strategy... 

Examples:
• Create a momentum-based portfolio focusing on high-growth tech stocks
• Build a dividend-focused portfolio with low volatility 
• Design a value investing strategy targeting undervalued stocks

Press Ctrl+Enter to generate, Escape to clear`
              : `Describe your forecasting method...

Examples:
• Predict stock prices using technical analysis with moving averages and RSI indicators
• Generate price forecasts based on fundamental analysis and earnings trends
• Create predictions using momentum indicators and volume analysis

Press Ctrl+Enter to generate, Escape to clear`}
            className="w-full min-h-[120px] max-h-[200px] px-4 py-3 bg-[#0F1419] border border-[#4CC9F0]/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#4CC9F0] focus:ring-1 focus:ring-[#4CC9F0] transition-colors resize-none disabled:opacity-50 disabled:cursor-not-allowed"
            rows={4}
          />
          
          {/* Character Counter */}
          <div className="absolute bottom-2 right-2 text-xs text-gray-500">
            <span className={charactersRemaining < 50 ? "text-orange-400" : charactersRemaining < 20 ? "text-red-400" : "text-gray-500"}>
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

        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          <button
            ref={generateButtonRef}
            type="button"
            onClick={handleGenerate}
            disabled={loading || disabled || !isDescriptionValid || !!validationError}
            className="flex-1 bg-[#4CC9F0] hover:bg-[#4CC9F0]/90 disabled:bg-gray-600 disabled:cursor-not-allowed text-black font-medium py-3 px-6 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-[#4CC9F0] focus:ring-offset-2 focus:ring-offset-[#1F2E45]"
          >
{loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" opacity="0.25"/>
                  <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
                {mode === 'backtest' ? 'Generating Strategy...' : 'Generating Forecast...'}
              </span>
            ) : (
              mode === 'backtest' ? 'Generate Strategy' : 'Generate Forecast'
            )}
          </button>
          
          <button
            type="button"
            onClick={handleClear}
            disabled={loading}
            className="px-4 py-3 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-[#1F2E45]"
          >
            Clear
          </button>
        </div>
      </div>

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
              {generatedCode && (
                <button
                  type="button"
                  onClick={() => setShowPreview(!showPreview)}
                  className="mt-2 text-xs text-[#4CC9F0] hover:text-[#4CC9F0]/80 transition-colors"
                >
                  {showPreview ? "Hide" : "Show"} Generated Code
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Code Preview Section */}
      {generatedCode && showPreview && (
        <div className="bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#4CC9F0]/20">
            <h3 className="text-sm font-medium text-gray-300">Generated Code</h3>
            <button
              type="button"
              onClick={() => setShowPreview(false)}
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="Close code preview"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="p-4">
            <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap bg-[#0F1419] rounded-lg p-4 border border-[#4CC9F0]/10">
              <code>{generatedCode}</code>
            </pre>
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Help */}
      <div className="text-center">
        <p className="text-xs text-gray-500">
          <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Ctrl+Enter</kbd> to generate •{" "}
          <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Esc</kbd> to clear
        </p>
      </div>
    </div>
  );
}