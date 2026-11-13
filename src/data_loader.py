"""
Data Loader & Preprocessing Pipeline for Retail Demand Forecasting.
Supports custom CSV uploads and built-in sample retail data.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sample_retail_data.csv"
)

def load_retail_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads retail transaction/demand data from CSV and standardizes schema.
    """
    path = file_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Standardize column naming
    rename_map = {
        "sales": "demand",
        "units": "demand",
        "quantity": "demand",
        "item_id": "sku_id",
        "product_id": "sku_id",
        "store": "store_id",
        "price": "unit_price",
        "cost": "unit_cost"
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    if "demand" not in df.columns:
        raise ValueError("Dataset must contain a 'demand' or 'sales' column.")

    # Defaults for optional metadata
    if "is_promo" not in df.columns:
        df["is_promo"] = 0
    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    if "unit_price" not in df.columns:
        df["unit_price"] = 10.0
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["unit_price"] * 0.6
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 5
    if "lead_time_std" not in df.columns:
        df["lead_time_std"] = 1.0

    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return df

def get_unique_stores_and_skus(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns unique store IDs and SKU IDs."""
    stores = sorted(df["store_id"].dropna().unique().tolist())
    skus = sorted(df["sku_id"].dropna().unique().tolist())
    return stores, skus

def filter_series(df: pd.DataFrame, store_id: str, sku_id: str) -> pd.DataFrame:
    """Filters dataset for a specific store and SKU combination."""
    filtered = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].copy()
    filtered = filtered.sort_values("date").reset_index(drop=True)
    return filtered

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal and autoregressive lag features for analytics and modeling.
    """
    res = df.copy()
    res["day_of_week"] = res["date"].dt.dayofweek
    res["month"] = res["date"].dt.month
    res["day_of_year"] = res["date"].dt.dayofyear
    res["is_weekend"] = res["day_of_week"].isin([5, 6]).astype(int)
    res["quarter"] = res["date"].dt.quarter

    # Lags & rolling windows
    res["lag_1"] = res["demand"].shift(1)
    res["lag_7"] = res["demand"].shift(7)
    res["lag_14"] = res["demand"].shift(14)
    res["rolling_mean_7"] = res["demand"].shift(1).rolling(window=7, min_periods=1).mean()
    res["rolling_std_7"] = res["demand"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    res["rolling_mean_28"] = res["demand"].shift(1).rolling(window=28, min_periods=1).mean()

    return res

def train_test_split_temporal(
    df: pd.DataFrame,
    test_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time series into train and test sets strictly chronologically.
    """
    sorted_df = df.sort_values("date").reset_index(drop=True)
    cutoff_date = sorted_df["date"].max() - pd.Timedelta(days=test_days)
    train_df = sorted_df[sorted_df["date"] <= cutoff_date].copy()
    test_df = sorted_df[sorted_df["date"] > cutoff_date].copy()
    return train_df, test_df
