'use client'

interface FilterProps {
  label: string
  children: React.ReactNode
}

export default function Filter({ label, children }: FilterProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-gray-400">{label}</span>
      {children}
    </div>
  )
}