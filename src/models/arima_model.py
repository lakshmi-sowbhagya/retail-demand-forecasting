"""
ARIMA / SARIMAX Model Wrapper for Retail Demand Planning.
Captures autocorrelation, autoregression, and moving average residual patterns
with exogenous promotional variables and confidence intervals.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

class ARIMADemandModel:
    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 0, 1, 7),
        use_exog: bool = True
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.model_res = None
        self.exog_cols = []
        self.last_date = None
        self.is_fitted = False
        self.demand_cap = None
        self.demand_floor = 0.0

    def _extract_exog(self, df: pd.DataFrame):
        exog_candidates = ["is_promo", "discount_pct"]
        cols = [c for c in exog_candidates if c in df.columns]
        if cols and self.use_exog:
            self.exog_cols = cols
            return df[cols].astype(float).values
        return None

    def fit(self, train_df: pd.DataFrame):
        """Fits SARIMAX on demand series."""
        train_df = train_df.sort_values("date").reset_index(drop=True)
        y = pd.to_numeric(train_df["demand"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        train_df = train_df.loc[valid].reset_index(drop=True)
        y = y[valid]

        if len(y) < 3:
            raise ValueError("ARIMA requires at least 3 valid demand observations.")

        exog = self._extract_exog(train_df)

        # Short uploaded histories cannot reliably estimate weekly seasonal terms.
        # Use a non-seasonal model for short series to avoid unstable extrapolation.
        seasonal_order = self.seasonal_order if len(y) >= 28 else (0, 0, 0, 0)

        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            self.model_res = model.fit(disp=False, maxiter=200)
        except Exception:
            # Stable, simpler fallback for short or difficult series.
            model = SARIMAX(
                y,
                exog=exog,
                order=(1, 0, 0),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            self.model_res = model.fit(disp=False, maxiter=100)

        # Keep forecasts within a data-driven retail-demand range. This prevents
        # unstable ARIMA extrapolation from dominating the dashboard scale.
        demand_mean = float(np.mean(y))
        demand_std = float(np.std(y))
        observed_max = float(np.max(y))
        self.demand_cap = max(observed_max * 3.0, demand_mean + 4.0 * demand_std, 1.0)

        self.last_date = train_df["date"].max()
        self.is_fitted = True
        return self

    def predict(self, future_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """Generates out-of-sample forecast with prediction intervals."""
        if not self.is_fitted or self.model_res is None:
            raise RuntimeError("Model must be fitted before predict.")

        n_steps = len(future_df)
        exog_future = None
        if self.exog_cols:
            exog_future = future_df[self.exog_cols].astype(float).values

        forecast_res = self.model_res.get_forecast(steps=n_steps, exog=exog_future)
        yhat = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(future_df["date"].values)
        cap = self.demand_cap if self.demand_cap is not None else np.inf
        result["yhat"] = np.clip(np.asarray(yhat, dtype=float), 0.0, cap)
        result["yhat_lower"] = np.clip(np.asarray(conf_int)[:, 0], 0.0, cap)
        result["yhat_upper"] = np.clip(np.asarray(conf_int)[:, 1], 0.0, cap)

        # Keep confidence intervals ordered after clipping.
        result["yhat_lower"] = np.minimum(result["yhat_lower"], result["yhat"])
        result["yhat_upper"] = np.maximum(result["yhat_upper"], result["yhat"])

        return result
