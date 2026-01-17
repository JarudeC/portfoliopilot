/**
 * Portfolio allocation pie chart.
 * Displays stock weights as a donut chart with dynamic color generation.
 * Handles zero-weight stocks by displaying them in legend with neutral color.
 */
'use client'

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts'

interface PortfolioPieChartProps {
  weights: Record<string, number>
  title?: string
  height?: number
  showTitle?: boolean
}

export default function PortfolioPieChart({
  weights,
  title = "Portfolio Weights",
  height = 300,
  showTitle = true
}: PortfolioPieChartProps) {
  if (!weights || Object.keys(weights).length === 0) {
    return (
      <div className="bg-[#0d1b2a]/50 rounded-lg p-4 flex items-center justify-center" style={{ height }}>
        <span className="text-gray-400">No data available</span>
      </div>
    )
  }

  // Process all stock data, give zero-weight stocks tiny values for legend display
  const data = Object.entries(weights)
    .map(([name, value]) => ({
      name,
      value: Math.abs(value) > 0.000001 ? Math.abs(value) : 0.000001, // Give tiny value to zero stocks for legend
      isZeroWeight: Math.abs(value) <= 0.000001 // Track which ones are actually zero
    }))
    .sort((a, b) => b.value - a.value);

  // Adjust layout based on number of stocks
  const stockCount = data.length
  const isMany = stockCount > 5

  return (
    <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
      {showTitle && (
        <h5 className="text-sm font-medium mb-3">{title}</h5>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={85}
            innerRadius={38}
            stroke="#0d1b2a"
            strokeWidth={2}
            paddingAngle={1}
          >
            {data.map((item, i) => {
              // Extended color palette for up to 8 stocks
              const hues = [190, 200, 210, 220, 230, 240, 250, 260]
              const lightness = stockCount <= 4 ? 55 - i * 3 : 50 - i * 2 // Adjust lightness spacing
              return (
                <Cell
                  key={i}
                  fill={item.isZeroWeight ? '#8B9DC3' : `hsl(${hues[i % hues.length]} 70% ${lightness}%)`}
                  stroke="#0d1b2a"
                />
              )
            })}
          </Pie>
          <Tooltip
            formatter={(v: number) => [(v * 100).toFixed(1) + "%", "Weight"]}
            contentStyle={{
              background: "#1B263B",
              border: "none",
              color: "#E0E8F9",
            }}
            itemStyle={{ color: "#E0E8F9" }}
          />
          <Legend
            verticalAlign="middle"
            align="right"
            layout="vertical"
            iconType="circle"
            wrapperStyle={{
              fontSize: "0.75rem",
              lineHeight: "1.25rem",
              color: "#E0E8F9",
              paddingLeft: "8px",
              maxWidth: "120px"
            }}
            // Custom payload for color-matched legend entries
            {...{
              payload: data.map((item, index) => {
                const hues = [190, 200, 210, 220, 230, 240, 250, 260];
                const lightness = stockCount <= 4 ? 55 - index * 3 : 50 - index * 2;
                return {
                  value: item.name,
                  type: 'circle' as const,
                  color: item.isZeroWeight ? '#8B9DC3' : `hsl(${hues[index % hues.length]} 70% ${lightness}%)`
                };
              })
            }}
            formatter={(value: string) => {
              return value.length > 10 ? value.substring(0, 10) + "..." : value
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}