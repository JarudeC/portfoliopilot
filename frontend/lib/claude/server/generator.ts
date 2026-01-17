/**
 * Claude AI strategy generation orchestrator.
 * Coordinates prompt creation, code extraction, execution, and fallback handling.
 *
 * Main entry points:
 * - generatePortfolioWeights(): Full AI generation with fallback layers
 * - generateCodeOnly(): Generate code without execution (for user review)
 * - executeUserCode(): Execute user-provided code with security validation
 */

import { validateSecurity, type SecurityConfig } from '../execution/security';
import type { StockData, GenerationResult, DashboardParams } from '../core/types';
import { DEFAULT_FORECAST_DAYS } from '../core/constants';
import {
  createRigidPrompt,
  extractTypeScriptCode,
  stripTypeScriptTypes,
  validateAndFixCode,
  BACKTEST_FALLBACK_FUNCTION,
  FORECAST_FALLBACK_FUNCTION
} from './code-processing';
import { executeWithTimeout, generateFallbackResult } from './fallback';

/**
 * Generate portfolio weights or predictions using Claude AI.
 * Implements multi-layer fallback for robustness.
 *
 * Flow:
 * 1. Validate user prompt for security
 * 2. Call Claude API to generate strategy code
 * 3. Extract and validate the generated code
 * 4. Execute code with timeout protection
 * 5. Fall back through multiple strategies if any step fails
 *
 * @param userDescription - User's strategy description
 * @param stockData - Stock data for the strategy
 * @param mode - 'forecast' or 'backtest'
 * @param claudeApiCall - Optional function to call Claude API
 * @param securityConfig - Security configuration
 * @param forecastDays - Number of forecast days
 * @param dashboardParams - Parameters from dashboard UI
 * @returns Generation result with weights/predictions
 */
export async function generatePortfolioWeights(
  userDescription: string,
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  claudeApiCall?: (prompt: string) => Promise<string>,
  securityConfig?: SecurityConfig,
  forecastDays: number = DEFAULT_FORECAST_DAYS,
  dashboardParams?: DashboardParams
): Promise<GenerationResult> {
  const startTime = Date.now();

  // Handle empty stock data
  if (!stockData || stockData.length === 0) {
    const fallbackResult = generateFallbackResult([{ symbol: 'DEFAULT', price: 100 }], 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: 'Empty stock data',
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }

  // Layer 0: Security validation of user prompt
  const promptSecurity = validateSecurity(userDescription, undefined, securityConfig);
  if (!promptSecurity.overallValid) {
    const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: `Security validation failed: ${promptSecurity.promptValidation.blockedReason}`,
      fallbackUsed: true,
      executionTime: Date.now() - startTime,
      securityValidation: {
        promptValid: false,
        blockedReason: promptSecurity.promptValidation.blockedReason,
        riskLevel: promptSecurity.combinedRiskLevel
      }
    };
  }

  // Layer 1: Try Claude API generation
  if (claudeApiCall) {
    try {
      const prompt = createRigidPrompt(userDescription, mode);
      const response = await claudeApiCall(prompt);

      // Layer 2: Extract and clean response
      const extractedCode = extractTypeScriptCode(response, mode);

      // Layer 2.5: Security validation of generated code
      const codeSecurity = validateSecurity(userDescription, extractedCode, securityConfig);
      if (!codeSecurity.overallValid) {
        const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
        return {
          success: true,
          type: mode,
          ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
          error: `Generated code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime,
          securityValidation: {
            promptValid: true,
            codeValid: false,
            blockedReason: codeSecurity.codeValidation?.blockedReason,
            riskLevel: codeSecurity.combinedRiskLevel
          }
        };
      }

      // Layer 3: Validate and fix code
      const validation = validateAndFixCode(extractedCode, mode);
      let codeToExecute = validation.fixedCode || extractedCode;

      // Strip TypeScript types for JavaScript execution
      codeToExecute = stripTypeScriptTypes(codeToExecute);

      // Layer 4: Execute with timeout
      const execution = await executeWithTimeout(codeToExecute, stockData, mode, forecastDays, 3000, dashboardParams);

      if (execution.success && execution.result) {
        return {
          success: true,
          type: mode,
          ...(mode === 'backtest' ? { weights: execution.result as number[] } : { predictions: execution.result as any[] }),
          code: codeToExecute,
          fallbackUsed: false,
          executionTime: Date.now() - startTime,
          securityValidation: {
            promptValid: true,
            codeValid: true,
            riskLevel: 'none'
          }
        };
      }
    } catch (error) {
      // Fall through to fallback layers
    }
  }

  // Layer 5-9: Multiple fallback strategies
  for (let layer = 1; layer <= 5; layer++) {
    try {
      const fallbackResult = generateFallbackResult(stockData, layer, mode, forecastDays);

      if (mode === 'backtest' && (fallbackResult as number[]).length === stockData.length) {
        return {
          success: true,
          type: mode,
          weights: fallbackResult as number[],
          error: `Used fallback layer ${layer}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime
        };
      } else if (mode === 'forecast' && Array.isArray(fallbackResult) && fallbackResult.length > 0) {
        return {
          success: true,
          type: mode,
          predictions: fallbackResult as any[],
          error: `Used fallback layer ${layer}`,
          fallbackUsed: true,
          executionTime: Date.now() - startTime
        };
      }
    } catch (error) {
      continue;
    }
  }

  // Ultimate fallback (Layer 10)
  const ultimateFallback = generateFallbackResult(stockData, 1, mode, forecastDays);
  return {
    success: true,
    type: mode,
    ...(mode === 'backtest' ? { weights: ultimateFallback as number[] } : { predictions: ultimateFallback as any[] }),
    error: 'Used ultimate fallback',
    fallbackUsed: true,
    executionTime: Date.now() - startTime
  };
}

/**
 * Generate strategy code without executing it.
 * Useful for code review before execution.
 *
 * @param userDescription - User's strategy description
 * @param stockData - Stock data context
 * @param mode - 'forecast' or 'backtest'
 * @param claudeApiCall - Optional function to call Claude API
 * @param securityConfig - Security configuration
 * @param forecastDays - Number of forecast days
 * @returns Generated code or error
 */
export async function generateCodeOnly(
  userDescription: string,
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  claudeApiCall?: (prompt: string) => Promise<string>,
  securityConfig?: SecurityConfig,
  forecastDays: number = DEFAULT_FORECAST_DAYS
): Promise<{ success: boolean; code?: string; error?: string; securityValidation?: any }> {

  if (!stockData || stockData.length === 0) {
    return { success: false, error: 'Empty stock data' };
  }

  // Security validation of user prompt
  const promptSecurity = validateSecurity(userDescription, undefined, securityConfig);
  if (!promptSecurity.overallValid) {
    return {
      success: false,
      error: `Security validation failed: ${promptSecurity.promptValidation.blockedReason}`,
      securityValidation: {
        promptValid: false,
        blockedReason: promptSecurity.promptValidation.blockedReason,
        riskLevel: promptSecurity.combinedRiskLevel
      }
    };
  }

  // Try Claude API generation
  if (claudeApiCall) {
    try {
      const prompt = createRigidPrompt(userDescription, mode);
      const response = await claudeApiCall(prompt);

      // Extract and clean response
      const extractedCode = extractTypeScriptCode(response, mode);

      // Security validation of generated code
      const codeSecurity = validateSecurity(userDescription, extractedCode, securityConfig);
      if (!codeSecurity.overallValid) {
        return {
          success: false,
          error: `Generated code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
          securityValidation: {
            promptValid: true,
            codeValid: false,
            blockedReason: codeSecurity.codeValidation?.blockedReason,
            riskLevel: codeSecurity.combinedRiskLevel
          }
        };
      }

      // Validate and fix code
      const validation = validateAndFixCode(extractedCode, mode);
      const codeToReturn = validation.fixedCode || extractedCode;

      return {
        success: true,
        code: codeToReturn,
        securityValidation: {
          promptValid: true,
          codeValid: true,
          riskLevel: 'none'
        }
      };
    } catch (error) {
      return {
        success: false,
        error: `Code generation failed: ${error}`
      };
    }
  }

  // Fallback - return template code
  const fallbackFunction = mode === 'forecast' ? FORECAST_FALLBACK_FUNCTION : BACKTEST_FALLBACK_FUNCTION;
  return {
    success: true,
    code: fallbackFunction,
    error: 'Used fallback template'
  };
}

/**
 * Execute user-provided strategy code with security validation.
 *
 * @param code - User's strategy code
 * @param stockData - Stock data for execution
 * @param mode - 'forecast' or 'backtest'
 * @param forecastDays - Number of forecast days
 * @param dashboardParams - Dashboard parameters
 * @param securityConfig - Security configuration
 * @returns Execution result with weights/predictions
 */
export async function executeUserCode(
  code: string,
  stockData: StockData[],
  mode: 'forecast' | 'backtest',
  forecastDays: number = DEFAULT_FORECAST_DAYS,
  dashboardParams?: any,
  securityConfig?: SecurityConfig
): Promise<GenerationResult> {
  const startTime = Date.now();

  if (!stockData || stockData.length === 0) {
    const fallbackResult = generateFallbackResult([{ symbol: 'DEFAULT', price: 100 }], 1, mode, forecastDays);
    return {
      success: false,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: 'Empty stock data',
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }

  try {
    // Security validation of code before execution
    const codeSecurity = validateSecurity('', code, securityConfig);
    if (!codeSecurity.overallValid) {
      const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
      return {
        success: false,
        type: mode,
        ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
        error: `Code blocked by security: ${codeSecurity.codeValidation?.blockedReason}`,
        fallbackUsed: true,
        executionTime: Date.now() - startTime,
        securityValidation: {
          promptValid: true,
          codeValid: false,
          blockedReason: codeSecurity.codeValidation?.blockedReason,
          riskLevel: codeSecurity.combinedRiskLevel
        }
      };
    }

    // Validate and fix code
    const validation = validateAndFixCode(code, mode);
    let codeToExecute = validation.fixedCode || code;

    // Strip TypeScript types for JavaScript execution
    codeToExecute = stripTypeScriptTypes(codeToExecute);

    // Execute with timeout
    const execution = await executeWithTimeout(codeToExecute, stockData, mode, forecastDays, 3000, dashboardParams);

    if (execution.success && execution.result) {
      return {
        success: true,
        type: mode,
        ...(mode === 'backtest' ? { weights: execution.result as number[] } : { predictions: execution.result as any[] }),
        code: codeToExecute,
        fallbackUsed: false,
        executionTime: Date.now() - startTime,
        securityValidation: {
          promptValid: true,
          codeValid: true,
          riskLevel: 'none'
        }
      };
    } else {
      // Execution failed, use fallback
      const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
      return {
        success: true,
        type: mode,
        ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
        error: execution.error || 'Execution failed',
        fallbackUsed: true,
        executionTime: Date.now() - startTime
      };
    }
  } catch (error) {
    // Execution error, use fallback
    const fallbackResult = generateFallbackResult(stockData, 1, mode, forecastDays);
    return {
      success: true,
      type: mode,
      ...(mode === 'backtest' ? { weights: fallbackResult as number[] } : { predictions: fallbackResult as any[] }),
      error: `Execution error: ${error}`,
      fallbackUsed: true,
      executionTime: Date.now() - startTime
    };
  }
}
