'use client'

interface SelectProps {
  value: any
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  options: (string | number)[]
  className?: string
}

export default function Select({ value, onChange, options, className = "" }: SelectProps) {
  return (
    <select
      value={value}
      onChange={onChange}
      className={`select-dark w-full ${className}`}
    >
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  )
}