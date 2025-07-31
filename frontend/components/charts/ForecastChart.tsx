'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'

interface ForecastChartProps {
  historySeries: Array<{ date: string; price: number }>
  forecastSeries: Array<{ date: string; price: number }>
  title?: string
  height?: number
  showTitle?: boolean
}

export default function ForecastChart({
  historySeries,
  forecastSeries,
  title,
  height = 300,
  showTitle = true
}: ForecastChartProps) {
  const allData = [...historySeries, ...forecastSeries]
  const splitDate = historySeries.length > 0 ? historySeries[historySeries.length - 1].date : null
  const tickInterval = Math.max(1, Math.floor(allData.length / 4))

  if (allData.length === 0) {
    return (
      <div className="bg-[#0d1b2a]/50 rounded-lg p-4 flex items-center justify-center" style={{ height }}>
        <span className="text-gray-400">No data available</span>
      </div>
    )
  }

  return (
    <div className="bg-[#0d1b2a]/50 rounded-lg p-4">
      {showTitle && title && (
        <h5 className="text-sm font-medium mb-3">{title}</h5>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={allData} margin={{ top: 5, right: 18, bottom: 5, left: 0 }}>
          <XAxis
            dataKey="date"
            interval={tickInterval}
            tickFormatter={(d) => {
              const dt = new Date(d)
              return `${(dt.getMonth() + 1).toString().padStart(2, "0")}/${dt
                .getDate()
                .toString()
                .padStart(2, "0")}`
            }}
            stroke="#7C8BAC"
            fontSize={10}
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
            formatter={(v: number) => [v.toFixed(2), 'Price']}
            labelFormatter={(l) => l.slice(0, 10)}
          />
          <Line
            dataKey="price"
            type="monotone"
            stroke="#4CC9F0"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {splitDate && (
            <ReferenceLine
              x={splitDate}
              stroke="#FF6B6B"
              strokeDasharray="3 3"
              strokeWidth={2}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}