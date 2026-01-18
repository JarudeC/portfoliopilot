/**
 * Fallback strategy generation when Claude AI fails.
 * Provides multiple fallback layers with different allocation strategies.
 */

import type { StockData, ExecutionResult } from '../core/types';
import {
  DEFAULT_FORECAST_DAYS,
  TREND_UPWARD,
  TREND_VOLATILE_RANGE,
  TREND_SIDEWAYS_RANGE,
  TREND_DOWNWARD,
  TREND_RANDOM_RANGE,
} from '../core/constants';

/**
 * Execute strategy code with timeout protection.
 * Used for both AI-generated and fallback code execution.
 *
 * @param code - JavaScript code to execute
 * @param stockData - Stock data to pass to the function
 * @param mode - 'forecast' or 'backtest'
 * @param forecastDays - Number of forecast days (forecast mode only)
 * @param timeoutMs - Execution timeout in milliseconds
 * @param dashboardParams - Additional parameters from dashboard
 * @returns Execution result with weights or predictions
 */
export function executeWithTimeout(
  code: string,
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  forecastDays: number = DEFAULT_FORECAST_DAYS,
  timeoutMs: number = 3000,
  dashboardParams?: any
): Promise<ExecutionResult> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      resolve({ success: false, error: 'Execution timeout', timeout: true });
    }, timeoutMs);

    try {
      let wrappedCode: string;
      let result: any;

      if (mode === 'forecast') {
        wrappedCode = `
          ${code}
          return generatePredictions(stockData, ${forecastDays}, dashboardParams);
        `;
        const fn = new Function('stockData', 'dashboardParams', wrappedCode);
        result = fn(stockData, dashboardParams || {});
      } else {
        wrappedCode = `
          ${code}
          return calculateWeights(stockData, dashboardParams);
        `;
        const fn = new Function('stockData', 'dashboardParams', wrappedCode);
        result = fn(stockData, dashboardParams || {});
      }

      clearTimeout(timeout);

      if (mode === 'forecast') {
        // Validate forecast result
        if (!Array.isArray(result)) {
          resolve({ success: false, error: 'Forecast function did not return an array', timeout: false });
          return;
        }

        // Validate and clean prediction structure
        const predictions = result.map((p) => {
          if (!p || typeof p !== 'object') return null;
          return {
            date: p.date || new Date().toISOString().split('T')[0],
            price: typeof p.price === 'number' && isFinite(p.price) ? p.price : 1.0,
            confidence: typeof p.confidence === 'number' ? Math.max(0, Math.min(1, p.confidence)) : 0.8
          };
        }).filter(p => p !== null);

        if (predictions.length === 0) {
          resolve({ success: false, error: 'No valid predictions generated', timeout: false });
          return;
        }

        resolve({ success: true, result: predictions as any, timeout: false });

      } else {
        // Validate backtest result (weights)
        if (!Array.isArray(result)) {
          resolve({ success: false, error: 'Function did not return an array', timeout: false });
          return;
        }

        const weights = result.map(w => {
          const num = Number(w);
          return isNaN(num) || !isFinite(num) ? 0 : num;
        });

        // Adjust length if needed
        if (weights.length !== stockData.length) {
          const correctLength = stockData.length;
          const adjustedWeights = new Array(correctLength).fill(1.0 / correctLength);
          resolve({ success: true, result: adjustedWeights, timeout: false });
          return;
        }

        // Normalize weights to sum to 1.0 (safety net in case AI didn't normalize)
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);
        const normalizedWeights = totalWeight > 0
          ? weights.map(w => w / totalWeight)
          : new Array(weights.length).fill(1.0 / weights.length);

        resolve({ success: true, result: normalizedWeights, timeout: false });
      }

    } catch (error) {
      clearTimeout(timeout);
      resolve({ success: false, error: error?.toString() || 'Unknown execution error', timeout: false });
    }
  });
}

/**
 * Generate fallback results when AI generation fails.
 * Multiple layers provide different strategies for robustness.
 *
 * Backtest fallback layers:
 * 1. Equal weights
 * 2. Market cap weighted
 * 3. Price weighted
 * 4. Volume weighted
 * 5. Random weights
 *
 * Forecast fallback layers:
 * 1. Simple upward trend
 * 2. Volatile trend
 * 3. Sideways trend
 * 4. Downward trend
 * 5. Random walk
 *
 * @param stockData - Stock data for weight calculation
 * @param layer - Fallback layer number (1-5)
 * @param mode - 'forecast' or 'backtest'
 * @param forecastDays - Number of forecast days
 * @returns Weights array (backtest) or predictions array (forecast)
 */
export function generateFallbackResult(
  stockData: StockData[],
  layer: number,
  mode: 'forecast' | 'backtest',
  forecastDays: number = DEFAULT_FORECAST_DAYS
): number[] | any[] {

  if (mode === 'forecast') {
    // Generate fallback predictions
    const predictions = [];
    const startDate = new Date();
    const basePrice = stockData[0]?.price || 100;

    for (let i = 1; i <= forecastDays; i++) {
      const futureDate = new Date(startDate);
      futureDate.setDate(startDate.getDate() + i);

      let trend: number;
      switch (layer) {
        case 1: // Simple upward trend
          trend = TREND_UPWARD;
          break;
        case 2: // Volatile trend
          trend = 1 + (Math.random() - 0.5) * TREND_VOLATILE_RANGE;
          break;
        case 3: // Sideways trend
          trend = 1 + (Math.random() - 0.5) * TREND_SIDEWAYS_RANGE;
          break;
        case 4: // Downward trend
          trend = TREND_DOWNWARD;
          break;
        default: // Random walk
          trend = 1 + (Math.random() - 0.5) * TREND_RANDOM_RANGE;
      }

      const predictedPrice = basePrice * Math.pow(trend, i);

      predictions.push({
        date: futureDate.toISOString().split('T')[0],
        price: Math.round(predictedPrice * 100) / 100,
        confidence: Math.max(0.3, 0.9 - (i * 0.02))
      });
    }

    return predictions;

  } else {
    // Generate fallback weights for backtest
    const count = stockData.length || 1;

    switch (layer) {
      case 1: // Equal weights
        return new Array(count).fill(1.0 / count);

      case 2: // Market cap weighted
        const marketCaps = stockData.map(s => s.marketCap || 1);
        const totalCap = marketCaps.reduce((a, b) => a + b, 0);
        return totalCap > 0 ? marketCaps.map(cap => cap / totalCap) : new Array(count).fill(1.0 / count);

      case 3: // Price weighted
        const prices = stockData.map(s => s.price || 1);
        const totalPrice = prices.reduce((a, b) => a + b, 0);
        return totalPrice > 0 ? prices.map(price => price / totalPrice) : new Array(count).fill(1.0 / count);

      case 4: // Volume weighted
        const volumes = stockData.map(s => s.volume || 1);
        const totalVolume = volumes.reduce((a, b) => a + b, 0);
        return totalVolume > 0 ? volumes.map(vol => vol / totalVolume) : new Array(count).fill(1.0 / count);

      case 5: // Random weights (normalized)
        const randomWeights = new Array(count).fill(0).map(() => Math.random());
        const randomSum = randomWeights.reduce((a, b) => a + b, 0);
        return randomSum > 0 ? randomWeights.map(w => w / randomSum) : new Array(count).fill(1.0 / count);

      default: // Ultimate fallback
        return new Array(count).fill(1.0 / count);
    }
  }
}
