"""
Retail Inventory Optimization & Replenishment Planning Engine.
Applies stochastic inventory theory under demand and lead-time uncertainty:
- Dynamic Safety Stock (SS) calculation for target Cycle Service Levels
- Reorder Point (ROP) determination
- Economic Order Quantity (EOQ) optimization
- Multi-day inventory trajectory simulation & purchase order generator
- Supply chain cost tradeoff modeling (Holding Cost vs. Stockout Cost)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, List

class InventoryOptimizer:
    def __init__(
        self,
        unit_cost: float,
        unit_price: float,
        lead_time_days: float = 7.0,
        lead_time_std: float = 1.0,
        annual_holding_rate: float = 0.20,
        order_cost_fixed: float = 50.0,
        stockout_penalty_per_unit: float = 15.0,
        service_level: float = 0.95
    ):
        self.unit_cost = max(unit_cost, 0.01)
        self.unit_price = max(unit_price, self.unit_cost)
        self.lead_time_days = max(lead_time_days, 1.0)
        self.lead_time_std = max(lead_time_std, 0.0)
        self.annual_holding_rate = annual_holding_rate
        self.daily_holding_cost = (self.unit_cost * self.annual_holding_rate) / 365.25
        self.order_cost_fixed = order_cost_fixed
        self.stockout_penalty = stockout_penalty_per_unit
        self.service_level = service_level
        self.z_score = float(norm.ppf(service_level))

    def calculate_safety_stock(self, daily_demand_mean: float, daily_demand_std: float) -> float:
        """
        SS = Z * sqrt( L * sigma_D^2 + D^2 * sigma_L^2 )
        Accounts for both demand volatility and supplier lead time variance.
        """
        var_demand_during_lt = (self.lead_time_days * (daily_demand_std ** 2)) + \
                               ((daily_demand_mean ** 2) * (self.lead_time_std ** 2))
        sigma_ltd = np.sqrt(max(var_demand_during_lt, 0.001))
        safety_stock = self.z_score * sigma_ltd
        return float(np.round(safety_stock, 1))

    def calculate_reorder_point(self, daily_demand_mean: float, safety_stock: float) -> float:
        """
        ROP = (Lead Time * Average Daily Demand) + Safety Stock
        """
        rop = (self.lead_time_days * daily_demand_mean) + safety_stock
        return float(np.round(rop, 1))

    def calculate_eoq(self, annual_demand: float) -> float:
        """
        EOQ = sqrt((2 * Demand_Annual * Order_Cost) / Annual_Holding_Cost)
        """
        annual_holding_per_unit = self.unit_cost * self.annual_holding_rate
        if annual_holding_per_unit <= 0 or annual_demand <= 0:
            return 50.0
        eoq = np.sqrt((2.0 * annual_demand * self.order_cost_fixed) / annual_holding_per_unit)
        return float(np.round(max(eoq, 10.0), 1))

    def compute_policy_summary(self, forecast_demand: np.ndarray, forecast_std: np.ndarray) -> Dict[str, Any]:
        """
        Calculates recommended steady-state inventory parameters from forecast.
        """
        mean_d = float(np.mean(forecast_demand))
        std_d = float(np.mean(forecast_std))
        annual_d = mean_d * 365.25

        ss = self.calculate_safety_stock(mean_d, std_d)
        rop = self.calculate_reorder_point(mean_d, ss)
        eoq = self.calculate_eoq(annual_d)

        # Financial metrics
        working_capital_ss = ss * self.unit_cost
        avg_cycle_inventory = eoq / 2.0
        total_avg_inventory_units = avg_cycle_inventory + ss
        total_working_capital = total_avg_inventory_units * self.unit_cost
        annual_cogs = annual_d * self.unit_cost
        inventory_turns = annual_cogs / (total_working_capital + 1e-6)

        return {
            "service_level_target_pct": self.service_level * 100.0,
            "z_score": round(self.z_score, 3),
            "safety_stock_units": ss,
            "reorder_point_units": rop,
            "economic_order_qty_units": eoq,
            "safety_stock_working_capital_usd": round(working_capital_ss, 2),
            "total_working_capital_usd": round(total_working_capital, 2),
            "annual_inventory_turns": round(inventory_turns, 2),
            "avg_daily_demand": round(mean_d, 2),
            "daily_demand_std": round(std_d, 2),
        }

    def simulate_replenishment_schedule(
        self,
        forecast_df: pd.DataFrame,
        initial_stock: float = None,
        use_scenario: str = "yhat" # 'yhat', 'P10', 'P90'
    ) -> pd.DataFrame:
        """
        Simulates dynamic daily inventory balance, PO generation, and stockout events
        over the planning horizon.
        """
        dates = pd.to_datetime(forecast_df["date"]).values
        n_days = len(dates)

        if use_scenario in forecast_df.columns:
            demand = forecast_df[use_scenario].values
        else:
            demand = forecast_df["yhat"].values

        mean_d = float(np.mean(demand))
        std_d = float(np.std(demand))
        annual_d = mean_d * 365.25

        ss = self.calculate_safety_stock(mean_d, std_d)
        rop = self.calculate_reorder_point(mean_d, ss)
        order_qty = self.calculate_eoq(annual_d)

        if initial_stock is None:
            initial_stock = rop + (order_qty * 0.5)

        on_hand = np.zeros(n_days)
        receipts = np.zeros(n_days)
        orders_placed = np.zeros(n_days)
        lost_sales = np.zeros(n_days)
        inventory_position = np.zeros(n_days)

        current_stock = float(initial_stock)
        pipeline_orders: List[Dict[str, Any]] = []

        lt_int = int(round(self.lead_time_days))

        for t in range(n_days):
            # 1. Process arriving receipts
            arriving_today = 0.0
            remaining_orders = []
            for order in pipeline_orders:
                if order["arrival_day"] == t:
                    arriving_today += order["quantity"]
                else:
                    remaining_orders.append(order)
            pipeline_orders = remaining_orders

            current_stock += arriving_today
            receipts[t] = arriving_today

            # 2. Fulfill daily demand
            d_t = demand[t]
            if current_stock >= d_t:
                current_stock -= d_t
                lost_s = 0.0
            else:
                lost_s = d_t - current_stock
                current_stock = 0.0

            on_hand[t] = current_stock
            lost_sales[t] = lost_s

            # 3. Check inventory position = On Hand + On Order
            on_order = sum(o["quantity"] for o in pipeline_orders)
            pos = current_stock + on_order
            inventory_position[t] = pos

            # 4. Trigger replenishment order if pos <= ROP
            if pos <= rop:
                po_qty = float(round(order_qty))
                arrival_day = t + lt_int
                pipeline_orders.append({"arrival_day": arrival_day, "quantity": po_qty})
                orders_placed[t] = po_qty

        sim_df = pd.DataFrame({
            "date": dates,
            "forecast_demand": np.round(demand, 1),
            "stock_level": np.round(on_hand, 1),
            "orders_received": receipts,
            "orders_placed": orders_placed,
            "lost_sales_units": np.round(lost_sales, 1),
            "inventory_position": np.round(inventory_position, 1),
            "reorder_point": rop,
            "safety_stock": ss
        })

        return sim_df
