"""Enhanced ARIMA forecaster with automatic order selection.

* Uses AIC/BIC criteria for optimal order selection
* Implements proper statistical validation
* Robust error handling and fallback mechanisms
* Returns four parallel lists that match the payload FastAPI expects:
    history_dates, history_values, forecast_dates, forecast_values
"""

from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from .base import ForecastRequest, load_series


# ──────────────────────────────────────────────────────────────────────────
#  Helper functions for ARIMA order selection
# ──────────────────────────────────────────────────────────────────────────

def _determine_differencing(series: pd.Series, max_d: int = 2) -> int:
    """Determine optimal differencing order using ADF test."""
    try:
        # Test original series
        adf_stat, p_value, _, _, _, _ = adfuller(series.dropna(), autolag='AIC')
        if p_value <= 0.05:
            return 0  # Series is already stationary
        
        # Test with differencing
        for d in range(1, max_d + 1):
            diff_series = series.diff(d).dropna()
            if len(diff_series) < 10:
                break
            adf_stat, p_value, _, _, _, _ = adfuller(diff_series, autolag='AIC')
            if p_value <= 0.05:
                return d
        
        return 1  # Default to first difference
    except:
        return 1  # Safe fallback


def _select_best_arima_order(series: pd.Series, max_p: int = 3, max_q: int = 3) -> tuple:
    """Select best ARIMA order using AIC criterion."""
    # Determine differencing order
    d = _determine_differencing(series)
    print(f"Selected differencing order d={d}")
    
    best_aic = float('inf')
    best_order = (1, 1, 1)
    tested_orders = []
    
    # Grid search over p and q
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0 and d == 0:
                continue  # Skip (0,0,0) model
            
            try:
                order = (p, d, q)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = ARIMA(series, order=order)
                    fitted = model.fit(method="statespace")
                    
                    # Check for valid AIC and successful fit
                    if hasattr(fitted, 'aic') and np.isfinite(fitted.aic):
                        tested_orders.append((order, fitted.aic))
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = order
                            
            except Exception as e:
                continue
    
    print(f"Tested {len(tested_orders)} ARIMA orders, best: {best_order} (AIC={best_aic:.2f})")
    return best_order


def _validate_arima_model(fitted_model) -> bool:
    """Validate that the ARIMA model is reasonable."""
    try:
        # Check if model converged and has valid AIC
        if not hasattr(fitted_model, 'aic') or not np.isfinite(fitted_model.aic):
            return False
        
        # Test forecast capability - this is the most important check
        test_forecast = fitted_model.forecast(steps=1)
        if pd.isna(test_forecast).any() or not np.isfinite(test_forecast).all():
            return False
        
        # Check if we have fitted values (indicates successful fit)
        if not hasattr(fitted_model, 'fittedvalues') or len(fitted_model.fittedvalues) == 0:
            return False
        
        return True
    except:
        return False


# ──────────────────────────────────────────────────────────────────────────
#  Public API expected by main.py
# ──────────────────────────────────────────────────────────────────────────

def forecast(req: ForecastRequest) -> Tuple[List[str], List[float], List[str], List[float]]:
    """Fit an optimal ARIMA model using statistical criteria and produce forecasts."""
    
    try:
        # 1) Load data – returns trading‑day indexed Series
        y = load_series(req.ticker, req.start, req.end)
        
        if len(y) < 20:
            raise ValueError(f"Insufficient data: only {len(y)} observations. Need at least 20 for proper ARIMA modeling.")
        
        # Fix frequency issue - ensure the series has business day frequency
        if y.index.freq is None:
            # Infer frequency or set to business day
            try:
                y.index.freq = pd.infer_freq(y.index)
                if y.index.freq is None:
                    # Force business day frequency for financial data
                    y = y.asfreq('B', method='ffill')
            except:
                # If inference fails, resample to business days
                y = y.resample('B').last().dropna()
                
        print(f"Data frequency: {y.index.freq}, length: {len(y)}")
        
        # 2) Automatic order selection using AIC criterion
        print(f"Selecting optimal ARIMA order for {req.ticker}...")
        
        fitted = None
        best_order = None
        
        try:
            # First attempt: Use statistical criteria for order selection
            best_order = _select_best_arima_order(y, max_p=3, max_q=3)
            print(f"Selected ARIMA order: {best_order}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = ARIMA(y, order=best_order)
                fitted = model.fit(method="statespace")
                
                # Validate the fitted model
                if not _validate_arima_model(fitted):
                    fitted = None
                    print(f"Model validation failed for order {best_order}")
        
        except Exception as e:
            print(f"Auto-selection failed: {e}")
            fitted = None
        
        # 3) Fallback to robust manual selection if auto-selection fails
        if fitted is None:
            print("Falling back to manual order selection...")
            fallback_orders = [
                (1, 1, 1),  # Classic ARIMA
                (2, 1, 2),  # More complex
                (1, 1, 0),  # AR with differencing
                (0, 1, 1),  # MA with differencing
                (2, 1, 0),  # Higher order AR
                (0, 1, 2),  # Higher order MA
            ]
            
            for order in fallback_orders:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        model = ARIMA(y, order=order)
                        test_fitted = model.fit(method="statespace")
                        
                        if _validate_arima_model(test_fitted):
                            fitted = test_fitted
                            best_order = order
                            print(f"Fallback successful with order: {order}")
                            break
                except:
                    continue
        
        # 4) Last resort: Simple random walk if all else fails
        if fitted is None:
            print("Using random walk as last resort...")
            try:
                model = ARIMA(y, order=(0, 1, 0))
                fitted = model.fit(method="statespace")
                best_order = (0, 1, 0)
            except:
                # If even random walk fails, use naive forecast
                print("ARIMA fitting completely failed, using naive forecast...")
                hist_dates = y.index.strftime("%Y-%m-%d").tolist()
                hist_values = y.tolist()
                last_date = y.index[-1].date()
                fc_dates = pd.bdate_range(last_date + timedelta(days=1), periods=req.horizon).strftime("%Y-%m-%d").tolist()
                fc_values = [float(y.iloc[-1])] * req.horizon
                return hist_dates, hist_values, fc_dates, fc_values
        
        print(f"Final model order: {best_order}")
        
        # 5) Generate forecasts
        try:
            fc_res = fitted.get_forecast(steps=req.horizon)
            fc_values = fc_res.predicted_mean.tolist()
            
            # Get confidence intervals for additional validation
            conf_int = fc_res.conf_int()
            
            # Validate forecast results
            if any(pd.isna(fc_values)) or not all(np.isfinite(fc_values)):
                raise ValueError("Forecast contains invalid values")
                
        except Exception as e:
            print(f"Forecasting failed: {e}, using alternative approach...")
            # Alternative: use simulate or manual forecast steps
            try:
                # Try manual step-by-step forecasting
                fc_values = []
                current_series = y.copy()
                
                for step in range(req.horizon):
                    # Refit model with current data and forecast one step
                    temp_model = ARIMA(current_series, order=best_order)
                    temp_fitted = temp_model.fit(method="statespace")
                    next_val = temp_fitted.forecast(steps=1).iloc[0]
                    fc_values.append(float(next_val))
                    
                    # Add forecasted value to series for next iteration
                    next_date = current_series.index[-1] + pd.Timedelta(days=1)
                    # Ensure we use business days
                    while next_date.weekday() >= 5:  # Skip weekends
                        next_date += pd.Timedelta(days=1)
                    current_series = pd.concat([current_series, pd.Series([next_val], index=[next_date])])
                    
            except Exception as e2:
                print(f"Alternative forecasting also failed: {e2}, using trend-aware fallback...")
                # Last resort: use trend from recent values instead of flat line  
                recent_values = y.tail(min(10, len(y)//4))  # Use last 10 values or 25% of data
                if len(recent_values) >= 2:
                    # Calculate simple linear trend
                    x_vals = np.arange(len(recent_values))
                    y_vals = recent_values.values
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                    last_val = float(y.iloc[-1])
                    fc_values = [last_val + slope * (i + 1) for i in range(req.horizon)]
                else:
                    # Ultimate fallback: last value
                    fc_values = [float(y.iloc[-1])] * req.horizon
        
        # 6) Build final output
        hist_dates = y.index.strftime("%Y-%m-%d").tolist()
        hist_values = y.tolist()
        
        last_date = y.index[-1].date()
        fc_dates = pd.bdate_range(last_date + timedelta(days=1), periods=req.horizon).strftime("%Y-%m-%d").tolist()
        
        # Ensure forecast values are clean floats
        fc_values = [float(v) for v in fc_values]
        
        # Final validation - ensure all return values are proper lists
        if not isinstance(hist_dates, list) or not isinstance(hist_values, list):
            raise ValueError("History data is not in list format")
        if not isinstance(fc_dates, list) or not isinstance(fc_values, list):
            raise ValueError("Forecast data is not in list format")
        if len(hist_dates) != len(hist_values):
            raise ValueError(f"History data length mismatch: {len(hist_dates)} dates vs {len(hist_values)} values")
        if len(fc_dates) != len(fc_values):
            raise ValueError(f"Forecast data length mismatch: {len(fc_dates)} dates vs {len(fc_values)} values")
        if len(hist_dates) == 0:
            raise ValueError("No historical data returned")
        if len(fc_dates) == 0:
            raise ValueError("No forecast data returned")
        
        print(f"ARIMA forecast complete for {req.ticker} using order {best_order}")
        return hist_dates, hist_values, fc_dates, fc_values
        
    except Exception as e:
        # Convert any exception to a structured error that main.py can handle
        error_msg = f"ARIMA forecasting failed for {req.ticker}: {str(e)}"
        raise RuntimeError(error_msg) from e