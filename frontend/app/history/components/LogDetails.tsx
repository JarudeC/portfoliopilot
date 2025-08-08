// Detailed view component for individual training log records
'use client'

import { TrainingLog, BacktestResult, ForecastResult, BacktestMetrics } from '@/lib/types/training'
import LogCharts from './LogCharts'
import { MetricsTable, EquityChart, PortfolioPieChart, ForecastChart } from '@/components/charts'

interface LogDetailsProps {
  log: TrainingLog
}

export default function LogDetails({ log }: LogDetailsProps) {
  const renderParameters = () => {
    const params = log.parameters
    // Filter parameters by training type
    const allowedKeys = log.type === 'backtest' 
      ? ['eval_win', 'lookback', 'hist_days', 'tc']
      : ['history_days', 'forecast_days']
    
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(params)
          .filter(([key]) => allowedKeys.includes(key))
          .map(([key, value]) => (
            <div key={key} className="bg-[#1B263B] rounded p-3">
              <div className="text-xs text-gray-400 mb-1 capitalize">
                {key.replace(/_/g, ' ')}
              </div>
              <div className="text-sm font-medium">
                {Array.isArray(value) ? value.join(', ') : String(value)}
              </div>
            </div>
          ))}
      </div>
    )
  }

  const renderBacktestMetrics = () => {
    if (log.type !== 'backtest') return null
    
    // Check if metrics exist, otherwise try to calculate basic metrics from results
    let metrics = log.metrics
    
    // If no metrics but we have results, try to calculate basic metrics
    if (!metrics && log.results) {
      const results = log.results as any
      if (results.cumulative_returns && results.cumulative_returns.length > 0) {
        const finalReturn = results.cumulative_returns[results.cumulative_returns.length - 1]
        const returns = results.returns || []
        
        // Calculate basic metrics - values should be in decimal form for MetricsTable
        const totalReturn = (finalReturn - 1) // Keep as decimal, MetricsTable will format
        const volatility = returns.length > 0 ? 
          Math.sqrt(returns.reduce((sum: number, r: number) => sum + r * r, 0) / returns.length) * Math.sqrt(252) : 0
        
        metrics = {
          total_return: totalReturn,
          annual_return: totalReturn, // Simplified
          volatility: volatility,
          sharpe_ratio: volatility > 0 ? totalReturn / volatility : 0,
          max_drawdown: 0 // Would need more complex calculation
        }
      }
    }
    
    return (
      <div>
        <h4 className="text-sm font-semibold mb-3 text-cyan-300">Performance Metrics</h4>
        <MetricsTable metrics={metrics || {}} showTitle={false} />
      </div>
    )
  }

  const renderForecastChartsWithMetrics = () => {
    if (log.type !== 'forecast') return null

    const chartData = (log as any).charts
    const metrics = log.metrics as any
    
    if (!chartData) return null

    return (
      <div>
        <h4 className="text-sm font-semibold mb-3 text-cyan-300">Forecast Results</h4>
        <div className="grid grid-cols-1 gap-4">
          {Object.entries(chartData)
            .filter(([ticker, data]) => data && typeof data === 'object')
            .map(([ticker, data]: [string, any]) => {
              const { history, forecast } = data
              
              if (!history || !forecast) return null
              
              // Get metrics for this ticker, ensure MSE comes before MAE
              const tickerMetrics = metrics?.[ticker] || {}
              const orderedMetrics = {
                ...(tickerMetrics.mse !== undefined && { mse: tickerMetrics.mse }),
                ...(tickerMetrics.mae !== undefined && { mae: tickerMetrics.mae })
              }
              
              return (
                <div key={ticker} className="space-y-4">
                  {/* Chart */}
                  <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
                    <h5 className="text-sm font-semibold text-cyan-300 mb-3">{ticker}</h5>
                    <ForecastChart
                      historySeries={history}
                      forecastSeries={forecast}
                      height={200}
                      showTitle={false}
                    />
                  </div>
                  
                  {/* Metrics immediately below chart - matching dashboard pattern */}
                  {Object.keys(orderedMetrics).length > 0 && (
                    <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
                      <MetricsTable 
                        metrics={orderedMetrics} 
                        showTitle={false}
                        title={`${ticker} Forecast Accuracy`}
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        Algorithm: {log.model} | Data points: {history.length}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-6">
      <div>
        <h4 className="text-sm font-semibold mb-3 text-cyan-300">Parameters</h4>
        {renderParameters()}
      </div>

      {/* Add pie chart, graph and metrics components like dashboard */}
      {log.type === 'backtest' && log.results && (
        <div className="space-y-6">
          {/* Portfolio Pie Chart */}
          {((log.results as any).weights || log.parameters.weights) && (
            <div>
              <h4 className="text-sm font-semibold mb-3 text-cyan-300">Final Portfolio Weights</h4>
              <LogCharts log={log} chartType="pie" />
            </div>
          )}
          
          {/* Equity Curve */}
          <div>
            <h4 className="text-sm font-semibold mb-3 text-cyan-300">Equity Curve (PnL)</h4>
            <LogCharts log={log} chartType="equity" />
          </div>
          
          {/* Performance Metrics */}
          {renderBacktestMetrics()}
        </div>
      )}
      
      {log.type === 'forecast' && (
        <div className="space-y-6">
          {/* Forecast Charts with Metrics - matching dashboard pattern */}
          {renderForecastChartsWithMetrics()}
        </div>
      )}

      <div className="flex items-center justify-center pt-4 border-t border-[#1B263B] text-xs text-gray-400">
        <span>Training ID: {log.id}</span>
      </div>
    </div>
  )
}