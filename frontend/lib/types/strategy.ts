/**
 * Types for user-saved AI strategies.
 */

export interface Strategy {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  code_url: string;
  mode: 'backtest' | 'forecast';
  created_at: string;
  updated_at: string;
}

/**
 * Strategy with code loaded from storage.
 */
export interface HydratedStrategy extends Omit<Strategy, 'code_url'> {
  code: string;
}

/**
 * Input for creating a new strategy.
 */
export interface CreateStrategyInput {
  name: string;
  description?: string;
  code: string;
  mode: 'backtest' | 'forecast';
}

/**
 * Input for updating an existing strategy.
 */
export interface UpdateStrategyInput {
  name?: string;
  description?: string;
  code?: string;
}

/**
 * API response for strategy list.
 */
export interface StrategyListResponse {
  strategies: HydratedStrategy[];
}

/**
 * API response for single strategy.
 */
export interface StrategyResponse {
  strategy: HydratedStrategy;
}
