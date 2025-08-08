'use client'

interface MetricsTableProps {
  metrics: Record<string, number>
  title?: string
  showTitle?: boolean
}

export default function MetricsTable({
  metrics,
  title = "Performance Metrics",
  showTitle = true
}: MetricsTableProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
        {showTitle && (
          <h5 className="text-sm font-medium mb-3">{title}</h5>
        )}
        <div className="text-center py-4 text-gray-400">
          No metrics available
        </div>
      </div>
    )
  }

  const metricLabels: Record<string, string> = {
    // Backtest metrics
    Return: "Return",
    AnnualReturn: "Annual Ret.",
    DailyVol: "Daily Vol.",
    AnnualVol: "Annual Vol.",
    Sharpe: "Sharpe",
    Sortino: "Sortino",
    total_return: "Total Return",
    annual_return: "Annual Return",
    volatility: "Volatility",
    sharpe_ratio: "Sharpe Ratio",
    max_drawdown: "Max Drawdown",
    win_rate: "Win Rate",
    profit_factor: "Profit Factor",
    // Forecast metrics
    mse: "MSE",
    mae: "MAE"
  }

  const formatValue = (key: string, value: number): string => {
    if (key.toLowerCase().includes('rate') || key.toLowerCase().includes('return')) {
      return (value * 100).toFixed(2) + '%'
    }
    // Format MSE/MAE with appropriate precision
    if (key === 'mse' || key === 'mae') {
      return value.toFixed(2)
    }
    return value.toFixed(3)
  }

  return (
    <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
      {showTitle && (
        <h5 className="text-sm font-medium mb-3">{title}</h5>
      )}
      <div className="overflow-auto text-sm">
        <table className="w-full">
          <thead>
            <tr className="text-left text-gray-400">
              <th className="pb-2">Metric</th>
              <th className="pb-2 text-right">Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([key, value]) => (
              <tr key={key} className="border-t border-[#1B263B]">
                <td className="py-2 text-gray-300">
                  {metricLabels[key] || key}
                </td>
                <td className="py-2 text-right font-semibold">
                  {typeof value === "number" ? formatValue(key, value) : value ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}