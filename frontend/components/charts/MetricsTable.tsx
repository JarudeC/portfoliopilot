'use client'

interface MetricsTableProps {
  metrics: Record<string, string | number>
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

  // Define consistent order for metrics display (dashboard order)
  const metricOrder = [
    'Return',
    'AnnualReturn', 
    'DailyVol',
    'AnnualVol',
    'Sharpe',
    'Sortino',
    'mse',
    'mae'
  ]

  const metricLabels: Record<string, string> = {
    Return: "Return",
    AnnualReturn: "Annual Ret.",
    DailyVol: "Daily Vol.",
    AnnualVol: "Annual Vol.",
    Sharpe: "Sharpe",
    Sortino: "Sortino",
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
            {metricOrder
              .filter(key => key in metrics)
              .map((key) => {
                const value = metrics[key]
                return (
                  <tr key={key} className="border-t border-[#1B263B]">
                    <td className="py-2 text-gray-300">
                      {metricLabels[key] || key}
                    </td>
                    <td className="py-2 text-right font-semibold">
                      {typeof value === "number" ? formatValue(key, value) : value ?? "—"}
                    </td>
                  </tr>
                )
              })}
          </tbody>
        </table>
      </div>
    </div>
  )
}