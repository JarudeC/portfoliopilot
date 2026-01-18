// Database row structure (with storage URLs instead of raw data)
export interface TrainingLog {
  id: string;
  user_id: string;
  type: 'forecast' | 'backtest';
  timestamp: string;
  stocks: string[];
  model: string;
  parameters: Record<string, any>;
  results_url?: string;  // Path to results JSON in Supabase Storage
  charts_url?: string;   // Path to charts JSON in Supabase Storage
  metrics?: BacktestMetrics;
  status: 'completed' | 'failed' | 'in_progress';
  created_at: string;
}

// Hydrated log with actual data loaded from storage
export interface HydratedTrainingLog extends Omit<TrainingLog, 'results_url' | 'charts_url'> {
  results: ForecastResult | BacktestResult;
  charts?: ChartConfig[];
}

export interface ForecastResult {
  predictions: Array<{
    date: string;
    price: number;
    confidence?: number;
  }>;
  model_performance?: {
    mse?: number;
    mae?: number;
    r2_score?: number;
  };
}

export interface BacktestResult {
  returns: number[];
  cumulative_returns: number[];
  dates: string[];
  trades?: Array<{
    date: string;
    action: 'buy' | 'sell';
    symbol: string;
    quantity: number;
    price: number;
  }>;
}

export interface BacktestMetrics {
  total_return: number;
  annual_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate?: number;
  profit_factor?: number;
}

export interface ChartConfig {
  type: 'line' | 'candlestick' | 'bar';
  data: any[];
  options?: Record<string, any>;
}

// Input data for creating a training log (with actual data, will be uploaded to storage)
export interface CreateTrainingLogData {
  type: 'forecast' | 'backtest';
  stocks: string[];
  model: string;
  parameters: Record<string, any>;
  results: ForecastResult | BacktestResult;
  charts?: ChartConfig[] | Record<string, any> | null;  // Flexible chart data format
  metrics?: BacktestMetrics | Record<string, any> | null;
}