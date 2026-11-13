"""
Uncertainty Quantification & Monte Carlo Simulation Engine.
Translates point forecasts and forecast error distributions into:
- Quantile Demand Profiles (P10, P50, P90, P95, P99)
- Cumulative Lead-Time Demand Distributions
- Stochastic Monte Carlo paths for supply chain risk analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class DemandUncertaintySimulator:
    def __init__(self, yhat: np.ndarray, yhat_lower: np.ndarray, yhat_upper: np.ndarray, dates: pd.Series):
        self.yhat = np.asarray(yhat, dtype=float)
        self.dates = pd.to_datetime(dates).values
        self.n_steps = len(yhat)

        # Derive effective standard deviation from 95% confidence interval
        # Interval width = 2 * 1.96 * sigma
        interval_width = np.maximum(yhat_upper - yhat_lower, 0.1)
        self.sigma = interval_width / (2.0 * 1.96)

    def generate_quantiles(self) -> pd.DataFrame:
        """
        Calculates quantile trajectories: P10, P25, P50 (median), P75, P90, P95, P99.
        """
        z_scores = {
            "P10": -1.282,
            "P25": -0.674,
            "P50": 0.0,
            "P75": 0.674,
            "P90": 1.282,
            "P95": 1.645,
            "P99": 2.326
        }
        res = pd.DataFrame({"date": self.dates})
        for q_name, z in z_scores.items():
            demand_q = np.maximum(self.yhat + z * self.sigma, 0.0)
            res[q_name] = np.round(demand_q, 1)

        return res

    def run_monte_carlo(self, n_simulations: int = 1000, seed: int = 42) -> np.ndarray:
        """
        Runs n_simulations stochastic trajectories.
        Returns array of shape (n_simulations, n_steps).
        """
        np.random.seed(seed)
        # Generate Gaussian noise correlated with forecast variance
        noise = np.random.normal(0, 1, size=(n_simulations, self.n_steps))
        paths = self.yhat[np.newaxis, :] + noise * self.sigma[np.newaxis, :]
        return np.maximum(paths, 0.0)

    def lead_time_demand_distribution(
        self,
        lead_time_days: int = 7,
        n_simulations: int = 2000
    ) -> Dict[str, Any]:
        """
        Simulates total cumulative demand across supplier lead time window.
        Returns summary statistics: Mean, Std, P10, P50, P90, P95, P99.
        """
        window = min(lead_time_days, self.n_steps)
        paths = self.run_monte_carlo(n_simulations=n_simulations)
        # Sum demand across the lead time window
        ltd_sims = np.sum(paths[:, :window], axis=1)

        return {
            "lead_time_days": window,
            "mean_ltd": float(np.mean(ltd_sims)),
            "std_ltd": float(np.std(ltd_sims)),
            "P10": float(np.percentile(ltd_sims, 10)),
            "P50": float(np.percentile(ltd_sims, 50)),
            "P90": float(np.percentile(ltd_sims, 90)),
            "P95": float(np.percentile(ltd_sims, 95)),
            "P99": float(np.percentile(ltd_sims, 99)),
            "raw_simulations": ltd_sims
        }
