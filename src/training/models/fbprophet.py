from prophet import Prophet
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WeatherProphet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Prophet(
            seasonality_mode=cfg.seasonality_mode,
            yearly_seasonality=cfg.yearly_seasonality,
            weekly_seasonality=cfg.weekly_seasonality,
            daily_seasonality=cfg.daily_seasonality,
            changepoint_prior_scale=cfg.changepoint_prior_scale
        )

    def fit(self, df: pd.DataFrame, target_column: str):
        """Prepares the DataFrame and fits the Prophet model."""
        # Prophet strictly requires 'ds' and 'y' column names
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df[target_column]
        })

        logger.info(f"Fitting Prophet model for target: {target_column}...")
        self.model.fit(prophet_df)
        return self.model
