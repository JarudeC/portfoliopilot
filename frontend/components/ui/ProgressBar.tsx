'use client'

interface ProgressBarProps {
  progress: number
  className?: string
  showPercentage?: boolean
}

export default function ProgressBar({
  progress,
  className = "",
  showPercentage = true
}: ProgressBarProps) {
  return (
    <div className={`relative h-4 rounded-full bg-[#1B263B] ${className}`}>
      <div
        className="h-full rounded-full bg-gradient-to-r from-[#3A86FF] to-[#4CC9F0] transition-[width] duration-300"
        style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
      />
      {showPercentage && (
        <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-[#E0E8F9]">
          {progress.toFixed(0)}%
        </span>
      )}
    </div>
  )
}