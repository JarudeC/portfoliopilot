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

  const data = Object.entries(weights).map(([name, value]) => ({
    name,
    value
  }))

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
            cx="40%" // leave room on the right for legend
            cy="50%"
            outerRadius={85}
            innerRadius={38}
            stroke="#0d1b2a"
            strokeWidth={2}
            paddingAngle={2}
          >
            {data.map((_, i) => {
              // monochrome-variant palette: cyan → indigo range
              const hues = [190, 200, 210, 220, 230, 240]
              return (
                <Cell
                  key={i}
                  fill={`hsl(${hues[i % hues.length]} 70% ${55 - i * 3}%)`}
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
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}