"""ARIMA model validation utilities.

Provides functions to validate that fitted ARIMA models are reasonable
and capable of producing valid forecasts.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_fitted_model(fitted_model) -> bool:
    """
    Validate that an ARIMA model is properly fitted and usable.

    This function performs multiple checks to ensure the fitted ARIMA model
    is valid and can produce reliable forecasts.

    Args:
        fitted_model: Fitted statsmodels ARIMA result object

    Returns:
        True if model passes all validation checks, False otherwise

    Validation checks performed:
        1. Model has valid AIC (not NaN/inf)
        2. Can produce a test forecast
        3. Test forecast values are finite (not NaN/inf)
        4. Has fitted values (indicates successful fit)
    """
    try:
        # Check 1: Valid AIC
        if not hasattr(fitted_model, 'aic') or not np.isfinite(fitted_model.aic):
            logger.debug("Model validation failed: invalid AIC")
            return False

        # Check 2: Can forecast
        test_forecast = fitted_model.forecast(steps=1)

        # Check 3: Forecast is finite
        if pd.isna(test_forecast).any() or not np.isfinite(test_forecast).all():
            logger.debug("Model validation failed: forecast contains NaN/inf")
            return False

        # Check 4: Has fitted values
        if not hasattr(fitted_model, 'fittedvalues') or len(fitted_model.fittedvalues) == 0:
            logger.debug("Model validation failed: no fitted values")
            return False

        logger.debug("Model validation passed")
        return True

    except Exception as e:
        logger.debug(f"Model validation failed with exception: {e}")
        return False
