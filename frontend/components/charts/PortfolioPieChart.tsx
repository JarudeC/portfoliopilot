/**
 * Portfolio allocation pie chart.
 * Displays stock weights as a donut chart with dynamic color generation.
 * Handles zero-weight stocks by displaying them in legend with neutral color.
 * Supports hover interaction: hovering on slice or legend dims all other elements.
 */
'use client'

import { useState } from 'react'
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
  // Track active stock by name (not index) for consistent matching
  const [activeStock, setActiveStock] = useState<string | null>(null)

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
      value: Math.abs(value) > 0.000001 ? Math.abs(value) : 0.000001,
      isZeroWeight: Math.abs(value) <= 0.000001
    }))
    .sort((a, b) => b.value - a.value);

  const stockCount = data.length

  // Color generation helper - uses name to ensure consistent colors
  const getColorByName = (name: string) => {
    const item = data.find(d => d.name === name)
    const index = data.findIndex(d => d.name === name)
    if (!item || item.isZeroWeight) return '#8B9DC3'
    const hues = [190, 200, 210, 220, 230, 240, 250, 260]
    const lightness = stockCount <= 4 ? 55 - index * 3 : 50 - index * 2
    return `hsl(${hues[index % hues.length]} 70% ${lightness}%)`
  }

  // Slice hover handlers - use stock name
  const onPieEnter = (_: unknown, index: number) => {
    const item = data[index]
    if (item) setActiveStock(item.name)
  }
  const onPieLeave = () => setActiveStock(null)

  // Legend hover handlers - use stock name
  const onLegendEnter = (legendData: { value?: string }) => {
    const stockName = legendData?.value
    if (stockName) setActiveStock(stockName)
  }
  const onLegendLeave = () => setActiveStock(null)

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
            onMouseEnter={onPieEnter}
            onMouseLeave={onPieLeave}
          >
            {data.map((item) => {
              const baseColor = getColorByName(item.name)
              const isActive = activeStock === null || activeStock === item.name
              return (
                <Cell
                  key={item.name}
                  fill={baseColor}
                  stroke="#0d1b2a"
                  style={{
                    opacity: isActive ? 1 : 0.3,
                    transition: 'opacity 0.2s ease-in-out',
                    cursor: 'pointer'
                  }}
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
            onMouseEnter={onLegendEnter}
            onMouseLeave={onLegendLeave}
            {...{
              payload: data.map((item) => ({
                value: item.name,
                type: 'circle' as const,
                color: getColorByName(item.name)
              }))
            }}
            formatter={(value) => {
              const stockName = String(value)
              const isActive = activeStock === null || activeStock === stockName
              const displayValue = stockName.length > 10 ? stockName.substring(0, 10) + "..." : stockName
              return (
                <span
                  style={{
                    opacity: isActive ? 1 : 0.3,
                    transition: 'opacity 0.2s ease-in-out',
                    cursor: 'pointer'
                  }}
                >
                  {displayValue}
                </span>
              )
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}