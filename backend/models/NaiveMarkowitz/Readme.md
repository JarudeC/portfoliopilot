# Naive Markowitz Portfolio Optimization

This project implements a naive version of the Markowitz Mean-Variance portfolio optimization framework with noise-injected expected returns to simulate prediction uncertainty. It allows backtesting across rolling windows with options for different levels of noise control via the `eta` parameter.

## Directory Structure

├── Naive_Markowitz.py # Core class implementing the Naive Markowitz optimizer <br>
├── Naive_pipeline.py # Script for executing rolling window simulations and saving outputs <br>
├── Naive_plot.py # Script for plotting portfolio PnL <br>
├── Data/ <br>
│ └── Database.csv # Input CSV with historical stock prices (rows: tickers, cols: dates) <br>
├── Saved_files/ # Directory where simulation results are saved <br>

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib

## Getting Started

1. **Prepare Data**:
   - Place `Database.csv` in the `Data/` directory.
   - File must have tickers as row indices and date strings as column headers (e.g. "d20110103").

2. **Run Simulation**:
   - Run `Naive_pipeline.py` to perform rolling window backtesting.
   - Adjust `lookback_window`, `evaluation_window`, `number_of_window`, `etas`, and `transaction_cost_rate` as needed.

3. **Plot Results**:
   - Run `Naive_plot.py` to visualize the cumulative profit and loss (PnL).

## Parameters

- `lookback_window`: Range of dates for training data, e.g., `[1, 75]`.
- `evaluation_window`: Number of days in each evaluation period, usually `5`.
- `eta`:
  - `0`: Use training set mean returns.
  - `1`: Use evaluation window mean returns.
  - `0 < eta < 1`: Add noise to simulate prediction uncertainty.
- `transaction_cost_rate`: Proportional cost per reallocation (default: `0.0001`).

## Output Files (saved in `Saved_files/`)

- `PnL_Naive_Markowitz_eta=...csv`: Cumulative PnL over all windows.
- `Overall_return_Naive_Markowitz_eta=...csv`: Daily returns over all periods.
- `Portfolio_value_Naive_Markowitz_eta=...csv`: Portfolio value progression.
- `Daily_PnL_Naive_Markowitz_eta=...csv`: Daily PnL values.
- `Turnovers_Naive_Markowitz_eta=...csv`: Turnover at each reallocation step.

## Notes

- Set the random seed for reproducibility.
- Correlation matrices are calculated using Pearson method and standardized data.
- Singular covariance matrices are handled with pseudo-inverse.
