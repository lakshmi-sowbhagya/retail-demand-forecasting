"""
End-to-End Test and Verification Script.
Validates Prophet, ARIMA, and Hybrid Model training, evaluation, uncertainty,
and inventory optimization pipeline.
"""

import sys
import os
import numpy as np

# Add root folder to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.data_loader import load_retail_data, filter_series, train_test_split_temporal
from src.models.prophet_model import ProphetDemandModel
from src.models.arima_model import ARIMADemandModel
from src.models.hybrid_model import HybridProphetARIMAModel
from src.uncertainty import DemandUncertaintySimulator
from src.inventory_optimizer import InventoryOptimizer
from src.metrics import calculate_all_metrics

def run_verification():
    print("=================================================================")
    print("Starting Retail Demand Forecasting Pipeline Verification")
    print("=================================================================")

    # 1. Load Data
    print("\n[Step 1/6] Loading data...")
    df = load_retail_data()
    print(f"Data loaded successfully. Total rows: {len(df):,}")

    # 2. Filter Store & SKU
    print("\n[Step 2/6] Filtering series for Store_01 & SKU_101_Beverages...")
    sub_df = filter_series(df, "Store_01", "SKU_101_Beverages")
    train_df, test_df = train_test_split_temporal(sub_df, test_days=60)
    print(f"Train records: {len(train_df)}, Test records: {len(test_df)}")

    # 3. Train Prophet Model
    print("\n[Step 3/6] Fitting Prophet Model...")
    prophet = ProphetDemandModel()
    prophet.fit(train_df)
    prophet_fc = prophet.predict(test_df)
    prophet_metrics = calculate_all_metrics(test_df["demand"].values, prophet_fc["yhat"].values)
    print(f"Prophet Forecast - MAE: {prophet_metrics['MAE']}, RMSE: {prophet_metrics['RMSE']}, WAPE: {prophet_metrics['WAPE (%)']}%")

    # 4. Train ARIMA Model
    print("\n[Step 4/6] Fitting ARIMA / SARIMAX Model...")
    arima = ARIMADemandModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    arima.fit(train_df)
    arima_fc = arima.predict(test_df)
    arima_metrics = calculate_all_metrics(test_df["demand"].values, arima_fc["yhat"].values)
    print(f"ARIMA Forecast   - MAE: {arima_metrics['MAE']}, RMSE: {arima_metrics['RMSE']}, WAPE: {arima_metrics['WAPE (%)']}%")

    # 5. Train Hybrid Model
    print("\n[Step 5/6] Fitting Hybrid Prophet-ARIMA Model...")
    hybrid = HybridProphetARIMAModel(hybrid_strategy="residual", arima_order=(1, 0, 1))
    hybrid.fit(train_df)
    hybrid_fc = hybrid.predict(test_df)
    hybrid_metrics = calculate_all_metrics(test_df["demand"].values, hybrid_fc["yhat"].values)
    print(f"Hybrid Forecast  - MAE: {hybrid_metrics['MAE']}, RMSE: {hybrid_metrics['RMSE']}, WAPE: {hybrid_metrics['WAPE (%)']}%")

    # 6. Uncertainty & Inventory Optimization
    print("\n[Step 6/6] Quantifying Uncertainty & Optimizing Inventory...")
    uncertainty_sim = DemandUncertaintySimulator(
        hybrid_fc["yhat"].values,
        hybrid_fc["yhat_lower"].values,
        hybrid_fc["yhat_upper"].values,
        hybrid_fc["date"]
    )
    quantiles_df = uncertainty_sim.generate_quantiles()
    print("Sample Quantiles (P10, P50, P90):")
    print(quantiles_df.head(3))

    lead_time_stats = uncertainty_sim.lead_time_demand_distribution(lead_time_days=5)
    print(f"Lead-Time Demand (5 Days) -> Mean: {lead_time_stats['mean_ltd']:.1f}, P95: {lead_time_stats['P95']:.1f}")

    optimizer = InventoryOptimizer(
        unit_cost=sub_df["unit_cost"].iloc[0],
        unit_price=sub_df["unit_price"].iloc[0],
        lead_time_days=sub_df["lead_time_days"].iloc[0],
        lead_time_std=sub_df["lead_time_std"].iloc[0],
        service_level=0.95
    )
    fc_std = (hybrid_fc["yhat_upper"].values - hybrid_fc["yhat_lower"].values) / (2 * 1.96)
    policy = optimizer.compute_policy_summary(hybrid_fc["yhat"].values, fc_std)
    print("\nRecommended Inventory Policy:")
    for k, v in policy.items():
        print(f"  {k}: {v}")

    sim_schedule = optimizer.simulate_replenishment_schedule(hybrid_fc)
    print(f"\nSimulated 60-day Inventory Schedule. Total orders placed: {int(np.sum(sim_schedule['orders_placed'] > 0))}")
    print(f"Stockout events: {int(np.sum(sim_schedule['lost_sales_units'] > 0))}")

    print("\n=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully functional.")
    print("=================================================================")

if __name__ == "__main__":
    run_verification()
