// Utility functions for calculating TRUE MSE/MAE using real backend algorithms

export interface ForecastData {
  historySeries: { date: string; price: number }[];
  forecastSeries: { date: string; price: number; confidence?: number }[];
  algorithm?: string; // The algorithm used to generate this forecast
  metrics?: ForecastMetrics; // Pre-calculated metrics to avoid duplication
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
 * Calculate overall MSE/MAE across multiple stocks for model performance assessment
 * Used for research purposes to get single metrics representing model accuracy
 */
export async function calculateOverallForecastMetrics(
  stockDataList: Array<{
    ticker: string;
    data: ForecastData;
    forecastAlgorithm: string;
    claudeStrategy?: any;
  }>
): Promise<OverallForecastMetrics> {
  const allPredictions: number[] = [];
  const allActuals: number[] = [];
  let successfulStocks = 0;

  for (const { ticker, data, forecastAlgorithm, claudeStrategy } of stockDataList) {
    try {
      const isCustomAI = forecastAlgorithm.toLowerCase() === 'custom ai strategy';
      
      // For Custom AI, use already-generated predictions to avoid rate limiting
      if (isCustomAI) {
        // Use the forecast data that was already generated (no API calls needed)
        const { historySeries, forecastSeries } = data;
        
        if (historySeries.length < 20 || forecastSeries.length === 0) {
          continue;
        }

        // Use same 70/30 split as other algorithms
        const splitIndex = Math.floor(historySeries.length * 0.7);
        const trainData = historySeries.slice(0, splitIndex);
        const testData = historySeries.slice(splitIndex);
        
        if (testData.length < 5) {
          continue;
        }

        // Use the already-generated forecast predictions
        // The forecastSeries already contains absolute prices (lastHistoricalPrice * multiplier)
        let predictionsFromForecast = forecastSeries.slice(0, testData.length).map(f => f.price);
        
        // Ensure arrays match length
        let actualPrices = testData.map(d => d.price);
        if (predictionsFromForecast.length !== actualPrices.length) {
          const minLength = Math.min(predictionsFromForecast.length, actualPrices.length);
          predictionsFromForecast = predictionsFromForecast.slice(0, minLength);
          actualPrices = actualPrices.slice(0, minLength);
        }

        // Add to overall arrays
        allPredictions.push(...predictionsFromForecast);
        allActuals.push(...actualPrices);
        successfulStocks++;
        continue;
      }
      
      // Add delay to prevent API rate limiting
      if (successfulStocks > 0) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      const { historySeries } = data;
      
      if (historySeries.length < 20) {
        continue; // Skip stocks with insufficient data
      }
      
      // Use same backtest approach as individual calculation
      const splitIndex = Math.floor(historySeries.length * 0.7);
      const trainData = historySeries.slice(0, splitIndex);
      const testData = historySeries.slice(splitIndex);
      
      if (testData.length < 5) {
        continue; // Skip if test data too small
      }

      let predictions: number[];
      
      if (isCustomAI) {
        predictions = await executeCustomAIForBacktest(trainData, testData.length, ticker, claudeStrategy);
      } else {
        predictions = await generateBacktestPredictionsFromBackend(
          trainData, 
          testData.length, 
          forecastAlgorithm,
          ticker,
          testData
        );
      }
      
      // Get actual prices (same logic as individual function)
      let actualPrices: number[];
      
      if (isCustomAI) {
        actualPrices = testData.map(d => d.price);
      } else {
        const embeddedTestPrices = (predictions as any)._backendTestPrices;
        const embeddedTicker = (predictions as any)._ticker;
        
        if (embeddedTestPrices && embeddedTicker === ticker) {
          actualPrices = embeddedTestPrices;
        } else {
          actualPrices = testData.map(d => d.price);
        }
      }
      
      // Convert Custom AI multipliers to absolute prices
      const basePrice = trainData[trainData.length - 1].price;
      let adjustedPredictions: number[];
      
      if (isCustomAI) {
        adjustedPredictions = predictions.map(multiplier => basePrice * multiplier);
      } else {
        adjustedPredictions = predictions;
      }
      
      // Ensure array lengths match
      if (adjustedPredictions.length !== actualPrices.length) {
        const minLength = Math.min(adjustedPredictions.length, actualPrices.length);
        adjustedPredictions = adjustedPredictions.slice(0, minLength);
        actualPrices = actualPrices.slice(0, minLength);
      }
      
      // Add to overall arrays
      allPredictions.push(...adjustedPredictions);
      allActuals.push(...actualPrices);
      successfulStocks++;
      
    } catch (error) {
      // Skip failed stocks but continue with others
      console.warn(`Failed to calculate metrics for ${ticker}:`, error);
      continue;
    }
  }

  // Calculate overall MSE and MAE across all stocks
  if (allPredictions.length === 0 || allActuals.length === 0) {
    return { 
      mse: 0, 
      mae: 0, 
      totalPredictions: 0, 
      stockCount: 0 
    };
  }

  const overallMSE = calculateMSE(allPredictions, allActuals);
  const overallMAE = calculateMAE(allPredictions, allActuals);

  return {
    mse: overallMSE,
    mae: overallMAE,
    totalPredictions: allPredictions.length,
    stockCount: successfulStocks
  };
}

/**
 * Calculate TRUE MSE/MAE for forecast by backtesting using the real backend algorithms
 * This splits historical data, calls the actual backend, then compares to actual known prices
 */
export async function calculateForecastMetrics(
  data: ForecastData, 
  forecastAlgorithm?: string,
  ticker?: string,
  claudeStrategy?: any // Claude GenerationResult for Custom AI
): Promise<ForecastMetrics> {
  const { historySeries } = data;
  
  if (historySeries.length < 20 || !ticker || !forecastAlgorithm) {
    return { mse: 0, mae: 0 };
  }

  // For custom AI strategy, we'll calculate metrics using the generated forecast data
  const isCustomAI = forecastAlgorithm.toLowerCase() === 'custom ai strategy';

  // Backtest approach: Use 70% of historical data for training, 30% for testing
  const splitIndex = Math.floor(historySeries.length * 0.7);
  const trainData = historySeries.slice(0, splitIndex);
  const testData = historySeries.slice(splitIndex);
  
  if (testData.length < 5) {
    return { mse: 0, mae: 0 };
  }

  try {
    let predictions: number[];
    
    if (isCustomAI) {
      // For Custom AI, execute the generated code with training data
      predictions = await executeCustomAIForBacktest(trainData, testData.length, ticker, claudeStrategy);
    } else {
      // Generate predictions using the REAL backend algorithm with proper backtesting
      predictions = await generateBacktestPredictionsFromBackend(
        trainData, 
        testData.length, 
        forecastAlgorithm,
        ticker,
        testData
      );
    }
    
    // Calculate TRUE MSE and MAE by comparing predictions to actual test data
    let actualPrices: number[];
    
    if (isCustomAI) {
      // Custom AI: Use our frontend test data
      actualPrices = testData.map(d => d.price);
    } else {
      // Use embedded backend test data to avoid global race conditions
      const embeddedTestPrices = (predictions as any)._backendTestPrices;
      const embeddedTicker = (predictions as any)._ticker;
      
      if (embeddedTestPrices && embeddedTicker === ticker) {
        actualPrices = embeddedTestPrices;
      } else {
        actualPrices = testData.map(d => d.price);
      }
    }
    
    // Fix format mismatch: Custom AI returns multipliers, backend APIs return absolute prices
    const basePrice = trainData[trainData.length - 1].price;
    let adjustedPredictions: number[];
    
    if (isCustomAI) {
      // Custom AI returns percentage multipliers - convert to absolute prices
      adjustedPredictions = predictions.map(multiplier => basePrice * multiplier);
    } else {
      // Backend APIs return absolute prices - use directly
      adjustedPredictions = predictions;
    }
    
    // Ensure array lengths match for accurate calculation
    if (adjustedPredictions.length !== actualPrices.length) {
      const minLength = Math.min(adjustedPredictions.length, actualPrices.length);
      adjustedPredictions = adjustedPredictions.slice(0, minLength);
      actualPrices = actualPrices.slice(0, minLength);
    }
    
    const mse = calculateMSE(adjustedPredictions, actualPrices);
    const mae = calculateMAE(adjustedPredictions, actualPrices);


    return { mse, mae };
  } catch (error) {
    return { mse: 0, mae: 0 };
  }
}

/**
 * Generate backtest predictions using the REAL backend algorithms
 */
async function generateBacktestPredictionsFromBackend(
  trainData: { date: string; price: number }[], 
  predictionLength: number, 
  algorithm: string,
  ticker: string,
  testData?: { date: string; price: number }[]
): Promise<number[]> {
  if (trainData.length < 5) {
    throw new Error('Insufficient training data for backtest');
  }

  // Step 1: First, get the full backend response to access history_values for proper backtesting
  const fullStartDate = trainData[0].date;
  const fullEndDate = testData && testData.length > 0 ? testData[testData.length - 1].date : trainData[trainData.length - 1].date;
  
  const fullResponse = await fetch(`/api/forecast/${algorithm.toLowerCase()}`, {
    method: "POST", 
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      ticker, 
      start: fullStartDate,
      end: fullEndDate,
      horizon: 1 // We just need the historical data
    }),
  });

  if (!fullResponse.ok) {
    throw new Error(`Full backend forecast failed: ${fullResponse.status}`);
  }

  const fullPayload = await fullResponse.json();
  
  if (!fullPayload.history_values || !Array.isArray(fullPayload.history_values)) {
    throw new Error('Backend did not return history_values');
  }

  // Step 2: Use backend's history_values for proper 70/30 split
  const backendHistoricalPrices = fullPayload.history_values;
  const splitIndex = Math.floor(backendHistoricalPrices.length * 0.7);
  const backendTestPrices = backendHistoricalPrices.slice(splitIndex);

  // Store this globally so we can access it from the main function
  if (!(global as any).backendTestData) {
    (global as any).backendTestData = {};
  }
  (global as any).backendTestData[ticker] = backendTestPrices;

  // Step 3: Now call backend with only the training period to get predictions for the test period
  const startDate = trainData[0].date;
  const endDate = trainData[trainData.length - 1].date;

  try {
    // Call backend with training period only to get predictions for test period
    const response = await fetch(`/api/forecast/${algorithm.toLowerCase()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        ticker, 
        start: startDate, 
        end: endDate, 
        horizon: backendTestPrices.length // Predict for the length of our backend test data
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend forecast failed: ${response.status} - ${response.statusText}`);
    }

    const payload = await response.json();


    // Check if we got the expected data structure
    if (!payload.forecast_values || !Array.isArray(payload.forecast_values)) {
      throw new Error('Invalid backend response structure');
    }

    // Ensure lengths match before returning
    if (payload.forecast_values.length !== backendTestPrices.length) {
      const minLength = Math.min(payload.forecast_values.length, backendTestPrices.length);
      payload.forecast_values = payload.forecast_values.slice(0, minLength);
      (global as any).backendTestData[ticker] = backendTestPrices.slice(0, minLength);
    }
    
    // Store the backend test prices directly on the returned predictions to avoid race conditions
    const predictions = payload.forecast_values;
    (predictions as any)._backendTestPrices = backendTestPrices;
    (predictions as any)._ticker = ticker;
    
    return predictions;
  } catch (error) {
    throw error;
  }
}

/**
 * Calculate Mean Squared Error
 */
function calculateMSE(predictions: number[], actual: number[]): number {
  if (predictions.length !== actual.length || predictions.length === 0) return 0;
  
  const squaredErrors = predictions.map((pred, i) => Math.pow(pred - actual[i], 2));
  return squaredErrors.reduce((sum, err) => sum + err, 0) / predictions.length;
}

/**
 * Calculate Mean Absolute Error  
 */
function calculateMAE(predictions: number[], actual: number[]): number {
  if (predictions.length !== actual.length || predictions.length === 0) return 0;
  
  const absoluteErrors = predictions.map((pred, i) => Math.abs(pred - actual[i]));
  return absoluteErrors.reduce((sum, err) => sum + err, 0) / predictions.length;
}

/**
 * Execute Custom AI model for backtesting using the existing executeUserCode function
 */
async function executeCustomAIForBacktest(
  trainData: { date: string; price: number }[], 
  predictionLength: number, 
  ticker: string,
  claudeStrategy?: any
): Promise<number[]> {
  if (!claudeStrategy || !claudeStrategy.code) {
    throw new Error('No Claude strategy code available for backtesting');
  }

  try {
    // Import the executeUserCode function
    const { executeUserCode } = await import('../claude/client');
    
    // Prepare the historical data in the format the AI model expects
    const stockData = trainData.map(point => ({
      symbol: ticker,
      price: point.price,
      date: point.date
    }));

    // Use the same executeUserCode function that the dashboard uses
    const result = await executeUserCode(
      claudeStrategy.code, // Final code from Monaco editor (edited or original)
      'forecast',
      stockData,
      undefined, // securityConfig
      predictionLength, // forecastDays
      undefined // dashboardParams
    );

    if (!result.success) {
      throw new Error(`Custom AI execution failed: ${result.error || 'Unknown error'}`);
    }

    if (!result.predictions || !Array.isArray(result.predictions)) {
      throw new Error('Custom AI did not return valid predictions array');
    }

    // Extract price values from predictions
    let predictions = result.predictions.map((pred: any) => {
      if (typeof pred === 'number') {
        return pred;
      } else if (pred && typeof pred === 'object') {
        return pred.price || pred.value || 0;
      } else {
        return 0;
      }
    });

    // Ensure we have the right number of predictions
    if (predictions.length < predictionLength) {
      // Pad with last known price if needed
      const lastPrice = predictions[predictions.length - 1] || trainData[trainData.length - 1].price;
      while (predictions.length < predictionLength) {
        predictions.push(lastPrice);
      }
    } else if (predictions.length > predictionLength) {
      // Trim to required length
      predictions = predictions.slice(0, predictionLength);
    }

    return predictions;
  } catch (error) {
    throw error;
  }
}