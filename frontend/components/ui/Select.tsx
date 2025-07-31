'use client'

interface SelectProps {
  value: any
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  options: (string | number)[]
  className?: string
}

export default function Select({ value, onChange, options, className = "" }: SelectProps) {
  return (
    <div className={`relative inline-block w-full ${className}`}>
      <select
        value={value}
        onChange={onChange}
        className="select-dark appearance-none pr-8 w-full"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      <svg
        className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 text-gray-400 transform -translate-y-1/2"
        viewBox="0 0 20 20"
        fill="none"
        stroke="currentColor"
      >
        <path d="M6 8l4 4 4-4" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </div>
  )
}