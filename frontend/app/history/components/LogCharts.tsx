'use client'

import { TrainingLog, BacktestResult, ForecastResult } from '@/lib/types/training'
import ForecastChart from '@/components/charts/ForecastChart'
import EquityChart from '@/components/charts/EquityChart'
import PortfolioPieChart from '@/components/charts/PortfolioPieChart'

interface LogChartsProps {
  log: TrainingLog
  chartType?: 'pie' | 'equity' | 'forecast' | 'all'
}

export default function LogCharts({ log, chartType = 'all' }: LogChartsProps) {
  const renderPieChart = () => {
    if (log.type !== 'backtest') return null
    const results = log.results as any
    
    // Check both results.weights and parameters.weights for backward compatibility
    const weights = results.weights || log.parameters.weights
    
    if (!weights) {
      return <div className="text-gray-400">No portfolio weights available</div>
    }

    return (
      <PortfolioPieChart weights={weights} height={200} showTitle={false} />
    )
  }

  const renderEquityChart = () => {
    if (log.type !== 'backtest') return null
    const results = log.results as BacktestResult
    
    // Check for nav data in charts or try to build from cumulative returns
    const navData = (log.charts as any)?.nav || (log.charts as any)?.equity_curve
    
    let equityData
    if (navData) {
      // If nav is stored as object, convert to array
      if (typeof navData === 'object' && !Array.isArray(navData)) {
        equityData = Object.entries(navData).map(([date, value]) => ({
          date,
          value: value as number
        }))
      } else if (Array.isArray(navData)) {
        equityData = navData
      }
    } else if (results.dates && results.cumulative_returns) {
      // Fallback to building from cumulative returns
      equityData = results.dates.map((date, i) => ({
        date,
        value: results.cumulative_returns[i]
      }))
    }
    
    if (!equityData || equityData.length === 0) {
      return <div className="text-gray-400">No equity data available</div>
    }

    return (
      <EquityChart data={equityData} height={224} showTitle={false} />
    )
  }

  const renderBacktestCharts = () => {
    if (log.type !== 'backtest') return null

    const results = log.results as BacktestResult
    
    if (!results.dates || !results.cumulative_returns) {
      return <div className="text-gray-400">No chart data available</div>
    }

    const equityData = results.dates.map((date, i) => ({
      date,
      value: results.cumulative_returns[i]
    }))

    return (
      <div className="space-y-6">
        <EquityChart data={equityData} title="Equity Curve" />
      </div>
    )
  }

  const renderForecastCharts = () => {
    if (log.type !== 'forecast') return null

    const results = log.results as ForecastResult
    
    if (!results.predictions || results.predictions.length === 0) {
      return <div className="text-gray-400">No chart data available</div>
    }

    // Get historical data from parameters if available
    const params = log.parameters
    const historicalData = params.history_dates && params.history_values ? 
      params.history_dates.map((date: string, i: number) => ({
        date,
        price: params.history_values[i]
      })) : []

    const forecastData = results.predictions.map(pred => ({
      date: pred.date,
      price: pred.price
    }))

    return (
      <ForecastChart
        historySeries={historicalData}
        forecastSeries={forecastData}
        title="Price Forecast"
      />
    )
  }

  // Render specific chart type if requested
  if (chartType === 'pie') {
    return renderPieChart()
  }
  
  if (chartType === 'equity') {
    return renderEquityChart()
  }
  
  if (chartType === 'forecast') {
    return renderForecastCharts()
  }

  // Default: render all charts
  return (
    <div>
      {log.type === 'backtest' && renderBacktestCharts()}
      {log.type === 'forecast' && renderForecastCharts()}
      
      {!log.results && (
        <div className="text-center py-8 text-gray-400">
          No chart data available for this training session
        </div>
      )}
    </div>
  )
}