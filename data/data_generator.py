"""
Synthetic Retail Data Generator for Demand Planning & Time-Series Forecasting.
Generates multi-store, multi-SKU retail transaction history with realistic:
- Macro trend growth
- Day-of-week weekly seasonality (weekend surges)
- Annual / holiday seasonality (Black Friday, Cyber Week, Christmas, Summer peaks)
- Promotional price discounts and sales lifts
- Supply lead times, unit costs, and inventory parameters
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

def generate_retail_dataset(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    stores = [
        {"store_id": "Store_01", "name": "Metro Supercenter", "traffic_mult": 1.4},
        {"store_id": "Store_02", "name": "Suburban Retail Hub", "traffic_mult": 1.0},
        {"store_id": "Store_03", "name": "Express Downtown", "traffic_mult": 0.75},
    ]

    skus = [
        {
            "sku_id": "SKU_101_Beverages",
            "category": "FMCG / Beverage",
            "base_demand": 120,
            "unit_price": 4.50,
            "unit_cost": 2.20,
            "lead_time_mean": 3,
            "lead_time_std": 0.5,
            "promo_lift": 1.6,
            "seasonality_type": "summer_peak"
        },
        {
            "sku_id": "SKU_202_Apparel",
            "category": "Fashion / Outerwear",
            "base_demand": 45,
            "unit_price": 49.99,
            "unit_cost": 22.00,
            "lead_time_mean": 7,
            "lead_time_std": 1.5,
            "promo_lift": 2.2,
            "seasonality_type": "winter_peak"
        },
        {
            "sku_id": "SKU_303_Electronics",
            "category": "Consumer Electronics",
            "base_demand": 20,
            "unit_price": 199.00,
            "unit_cost": 135.00,
            "lead_time_mean": 14,
            "lead_time_std": 3.0,
            "promo_lift": 2.8,
            "seasonality_type": "q4_holiday_peak"
        },
        {
            "sku_id": "SKU_404_PantryStaple",
            "category": "Groceries / Organic",
            "base_demand": 85,
            "unit_price": 8.99,
            "unit_cost": 4.80,
            "lead_time_mean": 4,
            "lead_time_std": 0.8,
            "promo_lift": 1.4,
            "seasonality_type": "steady"
        }
    ]

    all_rows = []

    for store in stores:
        for sku in skus:
            # Baseline parameters
            base = sku["base_demand"] * store["traffic_mult"]
            price = sku["unit_price"]
            cost = sku["unit_cost"]

            # Macro upward trend (e.g. 8% growth per year)
            trend = 1.0 + (0.08 * np.arange(n_days) / 365.25)

            # Weekly seasonality (higher Friday-Sunday)
            day_of_week = date_range.dayofweek.values
            dow_weights = np.array([0.85, 0.88, 0.92, 0.98, 1.25, 1.40, 1.20])
            weekly_seasonality = dow_weights[day_of_week]

            # Yearly seasonality
            day_of_year = date_range.dayofyear.values
            if sku["seasonality_type"] == "summer_peak":
                yearly_seasonality = 1.0 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            elif sku["seasonality_type"] == "winter_peak":
                yearly_seasonality = 1.0 - 0.30 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            elif sku["seasonality_type"] == "q4_holiday_peak":
                # Flat most of the year, massive surge in Nov-Dec (day 305 to 365)
                yearly_seasonality = np.ones(n_days)
                q4_mask = (day_of_year >= 305) & (day_of_year <= 360)
                yearly_seasonality[q4_mask] = 1.0 + 1.2 * np.exp(-0.5 * ((day_of_year[q4_mask] - 335) / 15) ** 2)
            else:
                yearly_seasonality = 1.0 + 0.10 * np.sin(2 * np.pi * day_of_year / 365.25)

            # Promotions (stochastic promotional events lasting 3-7 days, approx 8 times/year)
            promo_flag = np.zeros(n_days, dtype=int)
            promo_discount = np.zeros(n_days)
            i = 10
            while i < n_days - 10:
                if np.random.rand() < 0.035:
                    promo_duration = np.random.choice([3, 4, 5, 7])
                    discount = np.random.choice([0.15, 0.20, 0.25, 0.30])
                    end_idx = min(i + promo_duration, n_days)
                    promo_flag[i:end_idx] = 1
                    promo_discount[i:end_idx] = discount
                    i += promo_duration + 12  # Cool down period
                else:
                    i += 1

            # Holiday bumps (Thanksgiving, Black Friday, Christmas, New Year)
            holiday_multiplier = np.ones(n_days)
            for d_idx, dt in enumerate(date_range):
                # Black Friday / Cyber Monday (late November)
                if dt.month == 11 and 23 <= dt.day <= 29 and dt.dayofweek in [4, 0]:
                    holiday_multiplier[d_idx] = 1.8
                # Christmas week (Dec 20 - Dec 24)
                elif dt.month == 12 and 20 <= dt.day <= 24:
                    holiday_multiplier[d_idx] = 1.6
                # New Year's Eve
                elif dt.month == 12 and dt.day == 31:
                    holiday_multiplier[d_idx] = 1.5

            # Calculate deterministic expected demand
            expected_demand = base * trend * weekly_seasonality * yearly_seasonality * holiday_multiplier
            promo_multiplier = 1.0 + (sku["promo_lift"] - 1.0) * (promo_discount / 0.25) * promo_flag
            expected_demand = expected_demand * promo_multiplier

            # Add stochastic Poisson/Gaussian variance (demand uncertainty)
            noise = np.random.normal(0, np.sqrt(np.maximum(expected_demand, 1.0)) * 0.8)
            actual_demand = np.maximum(np.round(expected_demand + noise), 0).astype(int)

            # Selling price with discount
            current_price = np.round(price * (1.0 - promo_discount), 2)

            for d_idx in range(n_days):
                all_rows.append({
                    "date": date_range[d_idx],
                    "store_id": store["store_id"],
                    "store_name": store["name"],
                    "sku_id": sku["sku_id"],
                    "category": sku["category"],
                    "demand": actual_demand[d_idx],
                    "is_promo": promo_flag[d_idx],
                    "discount_pct": promo_discount[d_idx],
                    "unit_price": current_price[d_idx],
                    "unit_cost": cost,
                    "lead_time_days": sku["lead_time_mean"],
                    "lead_time_std": sku["lead_time_std"],
                    "holding_cost_rate": 0.20
                })

    df = pd.DataFrame(all_rows)
    df["revenue"] = df["demand"] * df["unit_price"]
    df["gross_margin"] = (df["unit_price"] - df["unit_cost"]) * df["demand"]
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sample_retail_data.csv")
    print(f"Generating synthetic retail dataset...")
    df = generate_retail_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Total rows: {len(df):,}, Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head())
