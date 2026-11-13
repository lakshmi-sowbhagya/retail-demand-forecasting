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
        y = train_df["demand"].astype(float).values
        exog = self._extract_exog(train_df)

        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_res = model.fit(disp=False, maxiter=100)
        except Exception:
            # Fallback to simpler (1,1,1) if seasonal estimation fails
            model = SARIMAX(
                y,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_res = model.fit(disp=False, maxiter=50)

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
        result["yhat"] = np.maximum(yhat, 0.0)
        result["yhat_lower"] = np.maximum(conf_int[:, 0], 0.0)
        result["yhat_upper"] = np.maximum(conf_int[:, 1], 0.0)

        return result
