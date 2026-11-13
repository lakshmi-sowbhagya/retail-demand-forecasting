"""
Streamlit Web Dashboard: Demand Forecasting under Uncertainty via Prophet/ARIMA-hybrid Models
Applied to Retail Demand Planning & Inventory Optimization
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure local modules are discoverable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    load_retail_data,
    get_unique_stores_and_skus,
    filter_series,
    engineer_features,
    train_test_split_temporal
)
from src.models.prophet_model import ProphetDemandModel
from src.models.arima_model import ARIMADemandModel
from src.models.hybrid_model import HybridProphetARIMAModel
from src.uncertainty import DemandUncertaintySimulator
from src.inventory_optimizer import InventoryOptimizer
from src.metrics import calculate_all_metrics

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-hybrid {
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_data(custom_file=None):
    if custom_file is not None:
        return load_retail_data(custom_file)
    return load_retail_data()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/warehouse.png", width=64)
    st.title("Demand Planning Engine")
    st.caption("Prophet / ARIMA / Hybrid Forecasting under Uncertainty")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload Custom Sales CSV (Optional)", type=["csv"])
    try:
        raw_df = get_cached_data(uploaded_file)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        raw_df = load_retail_data()

    stores, skus = get_unique_stores_and_skus(raw_df)

    selected_store = st.selectbox("🏬 Select Store Location", stores, index=0)
    selected_sku = st.selectbox("🏷️ Select Product / SKU", skus, index=0)

    st.markdown("---")
    st.subheader("⚙️ Horizon & Inventory Settings")
    test_days = st.slider("Forecast Evaluation Horizon (Days)", min_value=14, max_value=90, value=60, step=7)
    service_level = st.slider("Target Service Level (Cycle SLA)", min_value=0.80, max_value=0.99, value=0.95, step=0.01, format="%.2f")
    custom_lead_time = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=7, step=1)
    custom_order_cost = st.number_input("Fixed Order Setup Cost ($/PO)", min_value=10.0, max_value=500.0, value=50.0, step=10.0)

# Filter series
series_df = filter_series(raw_df, selected_store, selected_sku)
series_df = engineer_features(series_df)

unit_price = float(series_df["unit_price"].iloc[-1])
unit_cost = float(series_df["unit_cost"].iloc[-1])
lead_std = float(series_df["lead_time_std"].iloc[-1])

# Header Banner
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("🛒 Retail Demand Forecasting & Inventory Optimization")
    st.write(f"**Store:** `{selected_store}` | **SKU:** `{selected_sku}` | **Unit Cost:** `${unit_cost:.2f}` | **Selling Price:** `${unit_price:.2f}`")

with col_stat:
    st.markdown("<div style='text-align: right; margin-top: 15px;'><span class='badge-hybrid'>⚡ Hybrid Prophet-ARIMA</span></div>", unsafe_allow_html=True)

# Tabs
tab_overview, tab_models, tab_uncertainty, tab_inventory, tab_data = st.tabs([
    "📊 EDA & Demand Signals",
    "🤖 Multi-Model Forecast (Prophet vs ARIMA vs Hybrid)",
    "🎲 Uncertainty & Risk Cones (P10-P90)",
    "📦 Inventory & Replenishment Simulator",
    "📥 Data & Order Export"
])

# ----------------- TAB 1: EDA & DEMAND SIGNALS -----------------
with tab_overview:
    st.subheader("Executive Demand Overview")

    k1, k2, k3, k4, k5 = st.columns(5)
    total_sales = series_df["demand"].sum()
    avg_sales = series_df["demand"].mean()
    volatility = (series_df["demand"].std() / avg_sales) * 100.0
    promo_days = series_df["is_promo"].sum()
    total_rev = series_df["revenue"].sum()

    k1.metric("Total Historical Demand", f"{total_sales:,} units")
    k2.metric("Average Daily Demand", f"{avg_sales:.1f} units")
    k3.metric("Demand Volatility (CV)", f"{volatility:.1f}%")
    k4.metric("Promotional Days Active", f"{promo_days} days")
    k5.metric("Total Revenue Generated", f"${total_rev:,.2f}")

    # Time series demand plot
    st.markdown("#### Historical Sales Trajectory with Moving Averages")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=series_df["date"], y=series_df["demand"], mode="lines", name="Actual Daily Demand", line=dict(color="#94A3B8", width=1)))
    fig_hist.add_trace(go.Scatter(x=series_df["date"], y=series_df["rolling_mean_7"], mode="lines", name="7-Day Rolling Mean", line=dict(color="#2563EB", width=2)))
    fig_hist.add_trace(go.Scatter(x=series_df["date"], y=series_df["rolling_mean_28"], mode="lines", name="28-Day Rolling Trend", line=dict(color="#DC2626", width=2.5)))
    fig_hist.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Seasonality and Promotion Breakdown
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Day-of-Week Seasonality")
        dow_df = series_df.groupby("day_of_week")["demand"].mean().reset_index()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_df["day_name"] = [day_names[d] for d in dow_df["day_of_week"]]
        fig_dow = px.bar(dow_df, x="day_name", y="demand", color="demand", color_continuous_scale="Blues", labels={"demand": "Average Units Sold", "day_name": "Day"})
        fig_dow.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    with c_right:
        st.markdown("#### Promotional Uplift Effect")
        promo_impact = series_df.groupby("is_promo")["demand"].mean().reset_index()
        promo_impact["Type"] = promo_impact["is_promo"].map({0: "Regular Price", 1: "Promotional Discount"})
        fig_promo = px.bar(promo_impact, x="Type", y="demand", color="Type", color_discrete_sequence=["#64748B", "#10B981"], labels={"demand": "Average Units Sold"})
        fig_promo.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_promo, use_container_width=True)

# ----------------- MODEL TRAINING & INFERENCE PIPELINE -----------------
train_df, test_df = train_test_split_temporal(series_df, test_days=test_days)

@st.cache_resource
def train_and_evaluate_models(train_data: pd.DataFrame, test_data: pd.DataFrame):
    # 1. Prophet
    p_model = ProphetDemandModel()
    p_model.fit(train_data)
    p_fc = p_model.predict(test_data)

    # 2. ARIMA
    a_model = ARIMADemandModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    a_model.fit(train_data)
    a_fc = a_model.predict(test_data)

    # 3. Hybrid
    h_model = HybridProphetARIMAModel(hybrid_strategy="residual", arima_order=(1, 0, 1))
    h_model.fit(train_data)
    h_fc = h_model.predict(test_data)

    y_true = test_data["demand"].values
    metrics_p = calculate_all_metrics(y_true, p_fc["yhat"].values, unit_price, unit_cost)
    metrics_a = calculate_all_metrics(y_true, a_fc["yhat"].values, unit_price, unit_cost)
    metrics_h = calculate_all_metrics(y_true, h_fc["yhat"].values, unit_price, unit_cost)

    return (p_fc, a_fc, h_fc), (metrics_p, metrics_a, metrics_h)

with st.spinner("Training Prophet, ARIMA, and Hybrid models on historical data..."):
    (p_fc, a_fc, h_fc), (m_p, m_a, m_h) = train_and_evaluate_models(train_df, test_df)

# ----------------- TAB 2: MODEL COMPARISON -----------------
with tab_models:
    st.subheader("Model Evaluation & Forecast Comparison")
    st.write(f"Forecast evaluation period: **{len(test_df)} days** ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    # Forecast Visual Overlay
    fig_fc = go.Figure()
    # Actuals
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=test_df["demand"], mode="lines+markers", name="Actual Demand", line=dict(color="#0F172A", width=2)))
    # Prophet
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=p_fc["yhat"], mode="lines", name="Prophet Forecast", line=dict(color="#F59E0B", width=1.8, dash="dot")))
    # ARIMA
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=a_fc["yhat"], mode="lines", name="ARIMA / SARIMAX", line=dict(color="#8B5CF6", width=1.8, dash="dash")))
    # Hybrid
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=h_fc["yhat"], mode="lines", name="⚡ Prophet-ARIMA Hybrid", line=dict(color="#2563EB", width=2.8)))

    # Hybrid 95% Confidence Band
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=h_fc["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=h_fc["yhat_lower"], mode="lines", fill="tonexty", fillcolor="rgba(37, 99, 235, 0.15)", line=dict(width=0), name="Hybrid 95% Confidence Interval"))

    fig_fc.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_fc, use_container_width=True)

    # Leaderboard Table
    st.markdown("#### Model Performance Leaderboard")
    metrics_summary = pd.DataFrame([
        {"Model": "Prophet", **m_p},
        {"Model": "ARIMA / SARIMAX", **m_a},
        {"Model": "⚡ Prophet-ARIMA Hybrid", **m_h}
    ])

    # Highlight best in each column
    st.dataframe(metrics_summary.style.highlight_min(subset=["MAE", "RMSE", "WAPE (%)", "MAPE (%)", "Estimated Lost Sales Revenue ($)"], color="#DCFCE7"), use_container_width=True)

    # Residuals Analysis
    st.markdown("#### Residual Diagnostics (Actual - Forecast)")
    residuals = test_df["demand"].values - h_fc["yhat"].values
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        fig_res_time = px.line(x=test_df["date"], y=residuals, labels={"x": "Date", "y": "Residual Error"}, title="Residuals over Time (Hybrid Model)")
        fig_res_time.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res_time.update_layout(height=300)
        st.plotly_chart(fig_res_time, use_container_width=True)
    with c_res2:
        fig_res_dist = px.histogram(residuals, nbins=20, title="Residual Error Distribution", labels={"value": "Residual Error"}, color_discrete_sequence=["#3B82F6"])
        fig_res_dist.update_layout(height=300)
        st.plotly_chart(fig_res_dist, use_container_width=True)

# ----------------- TAB 3: UNCERTAINTY & RISK CONES -----------------
with tab_uncertainty:
    st.subheader("Demand Uncertainty Quantification & Monte Carlo Simulations")
    st.write("Quantiles allow supply chain planners to protect against stockout risk under demand variance.")

    uncertainty_engine = DemandUncertaintySimulator(h_fc["yhat"].values, h_fc["yhat_lower"].values, h_fc["yhat_upper"].values, h_fc["date"])
    quantiles_df = uncertainty_engine.generate_quantiles()

    # Fan chart
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=quantiles_df["date"], y=quantiles_df["P99"], mode="lines", line=dict(color="#E2E8F0", width=0.5), name="P99 (Extreme Surge)"))
    fig_fan.add_trace(go.Scatter(x=quantiles_df["date"], y=quantiles_df["P90"], mode="lines", fill="tonexty", fillcolor="rgba(191, 219, 254, 0.4)", line=dict(color="#93C5FD", width=1), name="P90 (High Demand)"))
    fig_fan.add_trace(go.Scatter(x=quantiles_df["date"], y=quantiles_df["P50"], mode="lines", line=dict(color="#1D4ED8", width=2.5), name="P50 (Median Expected)"))
    fig_fan.add_trace(go.Scatter(x=quantiles_df["date"], y=quantiles_df["P10"], mode="lines", fill="tonexty", fillcolor="rgba(191, 219, 254, 0.4)", line=dict(color="#93C5FD", width=1), name="P10 (Pessimistic Demand)"))
    fig_fan.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_fan, use_container_width=True)

    # Lead-Time Demand Simulation
    st.markdown(f"#### Cumulative Lead-Time Demand Distribution ({custom_lead_time} Days Window)")
    lt_stats = uncertainty_engine.lead_time_demand_distribution(lead_time_days=custom_lead_time)

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Expected Lead-Time Demand", f"{lt_stats['mean_ltd']:.1f} units")
    u2.metric("P50 (Median) Demand", f"{lt_stats['P50']:.1f} units")
    u3.metric("P90 Demand Scenario", f"{lt_stats['P90']:.1f} units")
    u4.metric("P99 Stress Scenario", f"{lt_stats['P99']:.1f} units")

    fig_ltd = px.histogram(lt_stats["raw_simulations"], nbins=35, title=f"Monte Carlo Simulated Total Demand over {custom_lead_time} Days Lead Time (2,000 Paths)", labels={"value": "Total Units Needed"}, color_discrete_sequence=["#6366F1"])
    fig_ltd.add_vline(x=lt_stats['P95'], line_dash="dash", line_color="red", annotation_text="P95 Service Threshold")
    fig_ltd.update_layout(height=320)
    st.plotly_chart(fig_ltd, use_container_width=True)

# ----------------- TAB 4: INVENTORY OPTIMIZER -----------------
with tab_inventory:
    st.subheader("Retail Inventory Optimization & Replenishment Simulator")

    optimizer = InventoryOptimizer(
        unit_cost=unit_cost,
        unit_price=unit_price,
        lead_time_days=custom_lead_time,
        lead_time_std=lead_std,
        order_cost_fixed=custom_order_cost,
        service_level=service_level
    )

    fc_std = (h_fc["yhat_upper"].values - h_fc["yhat_lower"].values) / (2 * 1.96)
    policy_summary = optimizer.compute_policy_summary(h_fc["yhat"].values, fc_std)

    # Top Policy Cards
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Safety Stock (SS)", f"{policy_summary['safety_stock_units']} units", help="Buffer stock held to absorb demand variance during supplier lead time")
    i2.metric("Reorder Point (ROP)", f"{policy_summary['reorder_point_units']} units", help="Inventory level that triggers a new purchase order")
    i3.metric("Economic Order Qty (EOQ)", f"{policy_summary['economic_order_qty_units']} units", help="Optimal purchase lot size balancing holding vs ordering costs")
    i4.metric("Annual Inventory Turns", f"{policy_summary['annual_inventory_turns']}x", help="Cost of goods sold divided by average working inventory")

    # Replenishment Simulation over Horizon
    sim_schedule = optimizer.simulate_replenishment_schedule(h_fc)

    st.markdown("#### Dynamic Stock Level & Purchase Order Simulation")
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=sim_schedule["date"], y=sim_schedule["stock_level"], mode="lines", name="On-Hand Inventory", line=dict(color="#10B981", width=2.5)))
    fig_sim.add_trace(go.Scatter(x=sim_schedule["date"], y=sim_schedule["inventory_position"], mode="lines", name="Inventory Position (On-Hand + On-Order)", line=dict(color="#3B82F6", width=1.5, dash="dot")))
    fig_sim.add_hline(y=policy_summary['reorder_point_units'], line_dash="dash", line_color="#F59E0B", annotation_text="Reorder Point (ROP)")
    fig_sim.add_hline(y=policy_summary['safety_stock_units'], line_dash="dash", line_color="#EF4444", annotation_text="Safety Stock Floor")

    # PO triggers
    po_placed = sim_schedule[sim_schedule["orders_placed"] > 0]
    if not po_placed.empty:
        fig_sim.add_trace(go.Scatter(x=po_placed["date"], y=po_placed["inventory_position"], mode="markers", marker=dict(color="#DC2626", size=10, symbol="triangle-up"), name="Purchase Order Placed"))

    fig_sim.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_sim, use_container_width=True)

    # Purchase Orders Table
    st.markdown("#### Automated Replenishment Purchase Order (PO) Recommendations")
    po_table = po_placed[["date", "orders_placed"]].copy()
    po_table.columns = ["Order Trigger Date", "Recommended Order Quantity (Units)"]
    po_table["Expected Delivery Date"] = pd.to_datetime(po_table["Order Trigger Date"]) + pd.Timedelta(days=custom_lead_time)
    po_table["Estimated Purchase Cost ($)"] = po_table["Recommended Order Quantity (Units)"] * unit_cost

    if po_table.empty:
        st.info("Current stock level satisfies the planning horizon without triggering a reorder.")
    else:
        st.dataframe(po_table.reset_index(drop=True), use_container_width=True)

# ----------------- TAB 5: DATA & EXPORT -----------------
with tab_data:
    st.subheader("Data Export & Automated Reports")
    st.write("Export detailed forecast outputs and purchase order schedules for downstream ERP/WMS systems (SAP, Oracle, NetSuite).")

    # Combine forecast into single export table
    export_df = pd.DataFrame({
        "date": test_df["date"].values,
        "actual_demand": test_df["demand"].values,
        "prophet_forecast": np.round(p_fc["yhat"].values, 1),
        "arima_forecast": np.round(a_fc["yhat"].values, 1),
        "hybrid_forecast": np.round(h_fc["yhat"].values, 1),
        "confidence_lower_95": np.round(h_fc["yhat_lower"].values, 1),
        "confidence_upper_95": np.round(h_fc["yhat_upper"].values, 1),
        "simulated_stock": sim_schedule["stock_level"].values,
        "replenishment_po_qty": sim_schedule["orders_placed"].values
    })

    st.dataframe(export_df.head(20), use_container_width=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Demand Forecast & Inventory Plan (CSV)",
            data=csv_data,
            file_name=f"demand_forecast_{selected_store}_{selected_sku}.csv",
            mime="text/csv"
        )
    with c_dl2:
        if not po_table.empty:
            po_csv = po_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Purchase Orders PO Schedule (CSV)",
                data=po_csv,
                file_name=f"replenishment_orders_{selected_store}_{selected_sku}.csv",
                mime="text/csv"
            )
