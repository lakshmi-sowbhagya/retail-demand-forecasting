"""
Prophet-ARIMA Hybrid Model for Retail Demand Forecasting.
Combines Facebook Prophet's non-linear trend, multi-frequency seasonality,
holiday effects, and external regressors with ARIMA/SARIMAX modeling of the
correlated residual error dynamics.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .prophet_model import ProphetDemandModel
from .arima_model import ARIMADemandModel

warnings.filterwarnings("ignore")

class HybridProphetARIMAModel:
    """
    Architecture:
    y_t = Prophet(t, X_t) + e_t
    e_t = ARIMA(p, d, q) + eta_t
    Final Forecast = y_hat_prophet + e_hat_arima
    """
    def __init__(
        self,
        hybrid_strategy: str = "residual",  # 'residual' or 'variance_weighted'
        prophet_kwargs: dict = None,
        arima_order: tuple = (1, 0, 1)
    ):
        self.hybrid_strategy = hybrid_strategy
        self.prophet_kwargs = prophet_kwargs or {}
        self.arima_order = arima_order

        self.prophet_model = ProphetDemandModel(**self.prophet_kwargs)
        self.arima_residual_model = None
        self.sarimax_direct_model = None
        self.ensemble_weights = (0.5, 0.5)
        self.residual_std = 1.0
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.sort_values("date").reset_index(drop=True)
        y = train_df["demand"].astype(float).values

        # 1. Fit Base Prophet Model
        self.prophet_model.fit(train_df)
        prophet_in_sample = self.prophet_model.predict(train_df)
        y_hat_prophet = prophet_in_sample["yhat"].values

        # 2. Extract Residuals
        residuals = y - y_hat_prophet
        self.residual_std = float(np.std(residuals))

        if self.hybrid_strategy == "residual":
            # Fit ARIMA on residuals
            try:
                self.arima_residual_model = ARIMA(residuals, order=self.arima_order).fit()
            except Exception:
                self.arima_residual_model = ARIMA(residuals, order=(1, 0, 0)).fit()

        elif self.hybrid_strategy == "variance_weighted":
            # Fit SARIMAX directly and compute optimal inverse-variance weights
            self.sarimax_direct_model = ARIMADemandModel()
            self.sarimax_direct_model.fit(train_df)
            sarimax_in_sample = self.sarimax_direct_model.predict(train_df)
            y_hat_arima = sarimax_in_sample["yhat"].values

            var_p = np.var(y - y_hat_prophet) + 1e-6
            var_a = np.var(y - y_hat_arima) + 1e-6
            # Inverse variance weighting: w1 = (1/var_p) / (1/var_p + 1/var_a)
            inv_p, inv_a = 1.0 / var_p, 1.0 / var_a
            w_prophet = inv_p / (inv_p + inv_a)
            w_arima = 1.0 - w_prophet
            self.ensemble_weights = (w_prophet, w_arima)

        self.is_fitted = True
        return self

    def predict(self, future_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Hybrid model must be fitted before predict.")

        future_df = future_df.sort_values("date").reset_index(drop=True)
        prophet_fc = self.prophet_model.predict(future_df)
        yhat_p = prophet_fc["yhat"].values

        n_steps = len(future_df)
        z = 1.96 if alpha == 0.05 else 1.645

        if self.hybrid_strategy == "residual":
            # Forecast residuals via ARIMA
            res_forecast = self.arima_residual_model.get_forecast(steps=n_steps)
            e_hat = res_forecast.predicted_mean
            e_se = res_forecast.se_mean

            hybrid_yhat = yhat_p + e_hat

            # Prophet confidence width converted to estimated standard error
            p_se = (prophet_fc["yhat_upper"].values - prophet_fc["yhat_lower"].values) / (2 * 1.96)
            combined_se = np.sqrt(p_se ** 2 + e_se ** 2)

            yhat_lower = hybrid_yhat - z * combined_se
            yhat_upper = hybrid_yhat + z * combined_se

        else: # variance_weighted
            arima_fc = self.sarimax_direct_model.predict(future_df, alpha=alpha)
            w_p, w_a = self.ensemble_weights
            hybrid_yhat = w_p * yhat_p + w_a * arima_fc["yhat"].values
            yhat_lower = w_p * prophet_fc["yhat_lower"].values + w_a * arima_fc["yhat_lower"].values
            yhat_upper = w_p * prophet_fc["yhat_upper"].values + w_a * arima_fc["yhat_upper"].values

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(future_df["date"].values)
        result["yhat"] = np.maximum(hybrid_yhat, 0.0)
        result["yhat_lower"] = np.maximum(yhat_lower, 0.0)
        result["yhat_upper"] = np.maximum(yhat_upper, 0.0)
        result["prophet_component"] = yhat_p
        if self.hybrid_strategy == "residual":
            result["residual_arima_component"] = e_hat

        return result
