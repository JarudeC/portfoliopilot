// Example integration of ClaudeStrategy component
// This shows how to use the component in a parent component

"use client";

import { useState } from "react";
import ClaudeStrategy from "./ClaudeStrategy";
import { ClaudeClientError, type GenerationResult, type StockData } from "../../lib/claude/client";

// Example parent component that integrates ClaudeStrategy
export default function StrategyGeneratorPage() {
  const [results, setResults] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Sample stock data (in a real app, this would come from props or API)
  const sampleStockData: StockData[] = [
    { symbol: 'AAPL', price: 150.00, marketCap: 2500000000000, volume: 50000000 },
    { symbol: 'GOOGL', price: 2800.00, marketCap: 1800000000000, volume: 1200000 },
    { symbol: 'MSFT', price: 300.00, marketCap: 2200000000000, volume: 30000000 },
    { symbol: 'TSLA', price: 800.00, marketCap: 800000000000, volume: 15000000 },
    { symbol: 'NVDA', price: 450.00, marketCap: 1100000000000, volume: 25000000 },
  ];

  // Handle successful strategy generation
  const handleStrategyGenerated = (result: GenerationResult) => {
    console.log('Strategy generated:', result);
    setResults(result);
    setError(null); // Clear any previous errors
  };

  // Handle errors from the component
  const handleError = (err: ClaudeClientError | Error) => {
    console.error('Strategy generation error:', err);
    if (err instanceof ClaudeClientError) {
      setError(`Error: ${err.message}`);
    } else {
      setError(`Unexpected error: ${err.message}`);
    }
    setResults(null); // Clear any previous results
  };

  return (
    <div className="min-h-screen bg-[#0F1419] py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-4">
            Portfolio Strategy Generator
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Use AI to generate custom portfolio allocation strategies based on your investment philosophy.
            Our system analyzes your description and creates optimized weight distributions.
          </p>
        </div>

        {/* ClaudeStrategy Component */}
        <ClaudeStrategy
          stockData={sampleStockData}
          onStrategyGenerated={handleStrategyGenerated}
          onError={handleError}
          className="mb-8"
        />

        {/* Results Display */}
        {results && (
          <div className="bg-[#1F2E45] border border-[#4CC9F0]/30 rounded-xl p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Generated Portfolio</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Portfolio Weights */}
              <div>
                <h4 className="text-lg font-medium text-gray-300 mb-3">Portfolio Weights</h4>
                <div className="space-y-2">
                  {sampleStockData.map((stock, index) => (
                    <div key={stock.symbol} className="flex justify-between items-center p-3 bg-[#0F1419] rounded-lg">
                      <span className="text-white font-medium">{stock.symbol}</span>
                      <span className="text-[#4CC9F0] font-mono">
                        {results.weights && ((results.weights[index] || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Strategy Metadata */}
              <div>
                <h4 className="text-lg font-medium text-gray-300 mb-3">Generation Details</h4>
                <div className="space-y-3">
                  <div className="p-3 bg-[#0F1419] rounded-lg">
                    <div className="text-sm text-gray-400">Execution Time</div>
                    <div className="text-white font-mono">{results.executionTime}ms</div>
                  </div>
                  
                  <div className="p-3 bg-[#0F1419] rounded-lg">
                    <div className="text-sm text-gray-400">Strategy Type</div>
                    <div className="text-white">
                      {results.fallbackUsed ? 'Fallback Strategy' : 'AI Generated'}
                    </div>
                  </div>

                  {results.securityValidation && (
                    <div className="p-3 bg-[#0F1419] rounded-lg">
                      <div className="text-sm text-gray-400">Security Status</div>
                      <div className="text-green-400">
                        ✓ Validated (Risk: {results.securityValidation.riskLevel})
                      </div>
                    </div>
                  )}

                  {results.error && (
                    <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                      <div className="text-sm text-orange-400">Notice</div>
                      <div className="text-orange-300 text-sm">{results.error}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex gap-3">
              <button 
                onClick={() => console.log('Export portfolio:', results)}
                className="bg-[#4CC9F0] hover:bg-[#4CC9F0]/90 text-black font-medium py-2 px-4 rounded-lg transition-colors"
              >
                Export Portfolio
              </button>
              <button 
                onClick={() => console.log('Backtest portfolio:', results)}
                className="border border-[#4CC9F0] text-[#4CC9F0] hover:bg-[#4CC9F0] hover:text-black font-medium py-2 px-4 rounded-lg transition-colors"
              >
                Run Backtest
              </button>
            </div>
          </div>
        )}

        {/* Global Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Usage Instructions */}
        <div className="mt-12 bg-[#1F2E45] border border-[#4CC9F0]/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-300">
            <div>
              <h4 className="font-medium text-white mb-2">Writing Strategy Descriptions</h4>
              <ul className="space-y-1 list-disc list-inside">
                <li>Be specific about your investment philosophy</li>
                <li>Mention risk tolerance (conservative, moderate, aggressive)</li>
                <li>Specify sectors or themes you prefer</li>
                <li>Include time horizon if relevant</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-white mb-2">Keyboard Shortcuts</h4>
              <ul className="space-y-1">
                <li><kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Ctrl+Enter</kbd> Generate strategy</li>
                <li><kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Escape</kbd> Clear form</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Alternative usage pattern for integration into existing components
export function IntegrateIntoExistingComponent() {
  return (
    <div className="my-8">
      <h2 className="text-xl font-bold text-white mb-4">Generate Custom Strategy</h2>
      
      <ClaudeStrategy
        stockData={[
          { symbol: 'SPY', price: 400 },
          { symbol: 'QQQ', price: 350 },
          { symbol: 'VTI', price: 200 },
        ]}
        onStrategyGenerated={(result) => {
          // Handle the generated strategy
          console.log('Generated portfolio weights:', result.weights);
          // You could update parent component state, make API calls, etc.
        }}
        onError={(error) => {
          // Handle errors
          console.error('Strategy generation failed:', error);
          // You could show toast notifications, update error state, etc.
        }}
      />
    </div>
  );
}