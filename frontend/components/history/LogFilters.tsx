'use client'

interface LogFiltersProps {
  filters: {
    type: 'all' | 'forecast' | 'backtest'
    search: string
    model: string
    dateFrom: string
    dateTo: string
  }
  onFiltersChange: (filters: any) => void
  availableModels: string[]
}

export default function LogFilters({ filters, onFiltersChange, availableModels }: LogFiltersProps) {
  const today = new Date().toISOString().split('T')[0]

  const updateFilter = (key: string, value: string) => {
    const newFilters = { ...filters, [key]: value }

    // Validate date constraints
    if (key === 'dateFrom' && newFilters.dateTo && value > newFilters.dateTo) {
      // If start date is after end date, clear end date
      newFilters.dateTo = ''
    }
    if (key === 'dateTo' && newFilters.dateFrom && value < newFilters.dateFrom) {
      // If end date is before start date, clear start date
      newFilters.dateFrom = ''
    }

    onFiltersChange(newFilters)
  }

  const clearFilters = () => {
    onFiltersChange({
      type: 'all',
      search: '',
      model: '',
      dateFrom: '',
      dateTo: ''
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Filters</h3>
        <button
          onClick={clearFilters}
          className="text-xs text-[#4CC9F0] hover:text-[#3A86FF] transition"
        >
          Clear All
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs text-gray-400 mb-2">Type</label>
          <select
            value={filters.type}
            onChange={(e) => updateFilter('type', e.target.value)}
            className="select-dark w-full"
          >
            <option value="all">All Types</option>
            <option value="forecast">Forecasts</option>
            <option value="backtest">Backtests</option>
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-2">Search</label>
          <input
            type="text"
            placeholder="Stock symbols or models..."
            value={filters.search}
            onChange={(e) => updateFilter('search', e.target.value)}
            className="select-dark w-full"
          />
        </div>

        {availableModels.length > 0 && (
          <div>
            <label className="block text-xs text-gray-400 mb-2">Model</label>
            <select
              value={filters.model}
              onChange={(e) => updateFilter('model', e.target.value)}
              className="select-dark w-full"
            >
              <option value="">All Models</option>
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="block text-xs text-gray-400 mb-2">Date Range</label>
          <div className="space-y-2">
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(e) => updateFilter('dateFrom', e.target.value)}
              max={filters.dateTo || today}
              className="select-dark w-full"
              placeholder="From"
            />
            <input
              type="date"
              value={filters.dateTo}
              onChange={(e) => updateFilter('dateTo', e.target.value)}
              min={filters.dateFrom || undefined}
              max={today}
              className="select-dark w-full"
              placeholder="To"
            />
          </div>
        </div>
      </div>
    </div>
  )
}