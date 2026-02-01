"""
Forecast metrics calculation functions (MSE, MAE).
Used by the batch forecast endpoint for backtesting validation.
"""

from typing import List


def calculate_mse(predictions: List[float], actuals: List[float]) -> float:
    """
    Calculate Mean Squared Error between predictions and actual values.

    Args:
        predictions: List of predicted values
        actuals: List of actual values

    Returns:
        MSE value, or 0.0 if inputs are invalid
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    return sum(squared_errors) / len(predictions)


def calculate_mae(predictions: List[float], actuals: List[float]) -> float:
    """
    Calculate Mean Absolute Error between predictions and actual values.

    Args:
        predictions: List of predicted values
        actuals: List of actual values

    Returns:
        MAE value, or 0.0 if inputs are invalid
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    absolute_errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return sum(absolute_errors) / len(predictions)
