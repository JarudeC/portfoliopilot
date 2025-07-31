'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

interface EquityChartProps {
  data: Array<{ date: string; value: number }>
  title?: string
  height?: number
  showTitle?: boolean
}

export default function EquityChart({
  data,
  title = "Equity Curve",
  height = 300,
  showTitle = true
}: EquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-[#0d1b2a]/50 rounded-lg p-3 flex items-center justify-center" style={{ height }}>
        <span className="text-gray-400">No data available</span>
      </div>
    )
  }

  return (
    <div className="bg-[#0d1b2a]/50 rounded-lg p-3">
      {showTitle && (
        <h4 className="text-sm font-semibold text-cyan-300 mb-2">{title}</h4>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 18, bottom: 5, left: 0 }}>
          <XAxis
            dataKey="date"
            tickFormatter={(d) => d.slice(2, 7)} // YY-MM format
            minTickGap={40}
            stroke="#7C8BAC"
            fontSize={12}
            tick={{ fill: "#7C8BAC" }}
            axisLine={{ stroke: "#7C8BAC" }}
          />
          <YAxis
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => v.toFixed(2)}
            stroke="#7C8BAC"
            fontSize={12}
          />
          <Tooltip
            contentStyle={{
              background: "#1B263B",
              border: "none",
              borderRadius: "4px",
              color: "#E0E8F9",
              fontSize: "12px",
            }}
            formatter={(v: number) => v.toFixed(4)}
            labelFormatter={(l) => l.slice(0, 10)}
          />
          <Line
            dataKey="value"
            type="monotone"
            stroke="#4CC9F0"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}