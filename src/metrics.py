"""
Comprehensive Forecasting and Inventory Evaluation Metrics.
Includes both statistical metrics (RMSE, MAE, MAPE, WAPE)
and Retail Business KPIs (Stockout units, Excess inventory, Lost sales revenue).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Protect against division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    WAPE is the retail industry gold standard: sum(|y - y_hat|) / sum(y).
    Unlike MAPE, it doesn't explode when daily demand is 0 or near 0.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_actual) * 100.0)

def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)

def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive indicates under-forecasting (y > y_hat), negative indicates over-forecasting."""
    return float(np.mean(y_true - y_pred))

def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_price: float = 10.0,
    unit_cost: float = 6.0,
    holding_cost_annual_rate: float = 0.20
) -> Dict[str, Any]:
    """
    Calculates full suite of statistical and financial supply-chain metrics.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    # Clip predictions to 0 (retail demand cannot be negative)
    y_p = np.maximum(y_p, 0.0)

    mae = mean_absolute_error(y_t, y_p)
    rmse = root_mean_squared_error(y_t, y_p)
    mape = mean_absolute_percentage_error(y_t, y_p)
    wape = weighted_absolute_percentage_error(y_t, y_p)
    smape = symmetric_mape(y_t, y_p)
    bias = forecast_bias(y_t, y_p)

    # Business calculations
    errors = y_t - y_p
    under_forecast_units = float(np.sum(np.maximum(errors, 0.0)))
    over_forecast_units = float(np.sum(np.maximum(-errors, 0.0)))

    unit_margin = max(unit_price - unit_cost, 0.0)
    lost_sales_revenue = under_forecast_units * unit_price
    lost_margin_dollars = under_forecast_units * unit_margin

    # Holding cost for over-forecasted inventory (assuming held for avg test horizon period)
    daily_holding_rate = holding_cost_annual_rate / 365.25
    excess_holding_cost = over_forecast_units * unit_cost * (daily_holding_rate * 30)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "SMAPE (%)": round(smape, 2),
        "Bias (Mean Error)": round(bias, 2),
        "Under-Forecast Units (Stockout Risk)": int(round(under_forecast_units)),
        "Over-Forecast Units (Excess Risk)": int(round(over_forecast_units)),
        "Estimated Lost Sales Revenue ($)": round(lost_sales_revenue, 2),
        "Estimated Lost Margin ($)": round(lost_margin_dollars, 2),
        "Estimated Excess Holding Cost ($)": round(excess_holding_cost, 2),
    }
