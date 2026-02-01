/**
 * Forecast metrics utilities.
 * For classical algorithms (ARIMA, LSTM, Autoformer), metrics come from the backend.
 * For Custom AI Strategy, metrics are calculated frontend-side using this module.
 */

export interface ForecastData {
  historySeries: { date: string; price: number }[];
  forecastSeries: { date: string; price: number; confidence?: number }[];
  algorithm?: string;
  metrics?: ForecastMetrics; // Backend-calculated for classical, frontend for Custom AI
}

export interface ForecastMetrics {
  mse: number;
  mae: number;
}

export interface OverallForecastMetrics {
  mse: number;
  mae: number;
  totalPredictions: number;
  stockCount: number;
}

/**
 * Get forecast metrics - returns backend metrics or calculates for Custom AI.
 * Uses 70/30 train/test split for backtesting validation.
 */
export async function calculateForecastMetrics(
  data: ForecastData,
  forecastAlgorithm?: string,
  _ticker?: string,      // Kept for API compatibility
  _claudeStrategy?: any  // Kept for API compatibility
): Promise<ForecastMetrics> {
  // If metrics already calculated by backend, return them
  if (data.metrics && (data.metrics.mse > 0 || data.metrics.mae > 0)) {
    return data.metrics;
  }

  // Only Custom AI reaches here - calculate frontend-side
  const isCustomAI = forecastAlgorithm?.toLowerCase() === 'custom ai strategy';
  if (!isCustomAI) {
    return { mse: 0, mae: 0 }; // Classical should have backend metrics
  }

  // Custom AI: use already-generated forecast data
  const { historySeries, forecastSeries } = data;
  if (historySeries.length < 20 || !forecastSeries.length) {
    return { mse: 0, mae: 0 };
  }

  // 70/30 split for backtesting
  const splitIndex = Math.floor(historySeries.length * 0.7);
  const testData = historySeries.slice(splitIndex);
  const minLen = Math.min(testData.length, forecastSeries.length);

  if (minLen < 5) {
    return { mse: 0, mae: 0 };
  }

  const actuals = testData.slice(0, minLen).map(d => d.price);
  const predictions = forecastSeries.slice(0, minLen).map(f => f.price);

  return {
    mse: calculateMSE(predictions, actuals),
    mae: calculateMAE(predictions, actuals)
  };
}

/**
 * Calculate Mean Squared Error
 */
function calculateMSE(predictions: number[], actual: number[]): number {
  if (predictions.length !== actual.length || predictions.length === 0) return 0;
  const sum = predictions.reduce((acc, p, i) => acc + (p - actual[i]) ** 2, 0);
  return sum / predictions.length;
}

/**
 * Calculate Mean Absolute Error
 */
function calculateMAE(predictions: number[], actual: number[]): number {
  if (predictions.length !== actual.length || predictions.length === 0) return 0;
  const sum = predictions.reduce((acc, p, i) => acc + Math.abs(p - actual[i]), 0);
  return sum / predictions.length;
}
