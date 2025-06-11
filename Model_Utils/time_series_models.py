import os
import yaml
import numpy as np
import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from prophet import Prophet
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml, append_yaml



# Setup logger
logger = setup_logger(filename="logs")

# ---------------- Strategy Interface ---------------- #
class ForecastStrategy(ABC):
    @abstractmethod
    def forecast_next(self, df: pd.DataFrame) -> float:
        pass


# ---------------- GBM Strategy ---------------- #
class GBMForecast(ForecastStrategy):

    @track_performance
    def forecast_next(self, df: pd.DataFrame) -> float:
        try:
            df = df.copy()
            df['log_return'] = np.log(df['nsei'] / df['nsei'].shift(1))
            mu = df['log_return'].mean()
            sigma = df['log_return'].std()
            S_t = df['nsei'].iloc[-1]
            Z = np.random.normal()
            S_t_plus_1 = S_t * np.exp((mu - 0.5 * sigma**2) + sigma * Z)
            return round(S_t_plus_1, 2)
        except CustomException as e:
            logger.error(f"GBM Forecast failed: {e}")
            return None


# ---------------- ARIMA Strategy ---------------- #
# ---------------- ARIMA Strategy ---------------- #
class ARIMAForecast(ForecastStrategy):
    def __init__(self, order=(5, 1, 0)):
        self.order = order

    def forecast_next(self, df: pd.DataFrame) -> float:
        try:
            df_arima = df[['date', 'nsei']].copy()
            df_arima['date'] = pd.to_datetime(df_arima['date'])
            df_arima.set_index('date', inplace=True)

            # Convert to PeriodIndex to suppress ARIMA warnings
            df_arima.index = pd.DatetimeIndex(df_arima.index).to_period('D')

            model = ARIMA(df_arima['nsei'], order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return round(forecast.iloc[0], 2)  # Use iloc to avoid KeyError
        except Exception as e:
            logger.error(f"ARIMA Forecast failed: {e}")
            return None



# ---------------- Prophet Strategy ---------------- #
class ProphetForecast(ForecastStrategy):

    @track_performance
    def forecast_next(self, df: pd.DataFrame) -> float:
        try:
            temp_df = df[['date', 'nsei']].rename(columns={'date': 'ds', 'nsei': 'y'})
            model = Prophet(daily_seasonality=True)
            model.fit(temp_df)
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            return round(forecast['yhat'].iloc[-1], 2)
        except CustomException as e:
            logger.error(f"Prophet Forecast failed: {e}")
            return None


# ---------------- VAR Strategy ---------------- #
class VARForecast(ForecastStrategy):

    @track_performance
    def forecast_next(self, df: pd.DataFrame) -> float:
        try:
            df = df[['date', 'nsei']].copy()
            df['day_of_week'] = df['date'].dt.dayofweek
            df.set_index('date', inplace=True)
            model = VAR(df.dropna())
            model_fit = model.fit(maxlags=15, ic='aic')
            forecast = model_fit.forecast(df.values[-model_fit.k_ar:], steps=1)
            return round(forecast[0][0], 2)
        except CustomException as e:
            logger.error(f"VAR Forecast failed: {e}")
            return None


# ---------------- Forecaster Context ---------------- #
class Forecaster:
    def __init__(self, strategy: ForecastStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ForecastStrategy):
        self.strategy = strategy

    def predict_next(self, df: pd.DataFrame) -> float:
        return self.strategy.forecast_next(df)

#---------------Average of all values of keys --------------------
def add_average_to_yaml(yaml_path: str, avg_key: str = "average_expected"):

    try:
        data = load_yaml(yaml_path)
        numeric_values = [float(v) for k,v in data.items() if k != "last_value"]
        avg = round(np.mean(numeric_values), 2)
        append_yaml(yaml_path, {avg_key: avg})
        logger.info(f"Appended {avg_key}: {avg} to {yaml_path}")
    except CustomException as e:
        logger.critical(f"Failed to compute and append average: {e}")

# ---------------- Main Function ---------------- #
@track_performance
def time_series_forecasts():
    try:
        df = pd.read_csv("Data/processed_data/preprocessed_data.csv", index_col=0)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f" final_df successfully read ito variable df ")
        
        prediction_dict = {}

        # Initialize forecaster
        forecaster = Forecaster(strategy=GBMForecast())
        gbm_pred = forecaster.predict_next(df)
        prediction_dict["GBM"] = gbm_pred
        mlflow.log_metric("GBM_next_day_price", gbm_pred)
        logger.info(f"GBM forecast successfully completed")

        # ARIMA
        forecaster.set_strategy(ARIMAForecast(order=(5, 1, 0)))
        arima_pred = forecaster.predict_next(df)
        prediction_dict["ARIMA"] = arima_pred
        mlflow.log_metric("ARIMA_next_day_price", arima_pred if arima_pred is not None else float('nan'))
        logger.info(f"ARIMA forecast successfully completed")

        # Prophet
        forecaster.set_strategy(ProphetForecast())
        prophet_pred = forecaster.predict_next(df)
        prediction_dict["Prophet"] = prophet_pred
        mlflow.log_metric("Prophet_next_day_price", prophet_pred if prophet_pred is not None else float('nan'))
        logger.info(f"Prophet forecast successfully completed")

        # VAR
        forecaster.set_strategy(VARForecast())
        var_pred = forecaster.predict_next(df)
        prediction_dict["VAR"] = var_pred
        mlflow.log_metric("VAR_next_day_price", var_pred if var_pred is not None else float('nan'))
        logger.info(f"VAR forecast successfully completed")

        # Logging the predictions
        logger.info("Next Day Price Predictions:")
        prediction_dict = {k: float(v) for k, v in prediction_dict.items()}
        for model, pred in prediction_dict.items():
            logger.info(f"{model}: {pred}")
            print(f"{model}: {pred}")

        # Save to YAML
        os.makedirs("Tuned_Model", exist_ok=True)
        output_path = os.path.join("Tuned_Model", "time_series_predictions.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(prediction_dict, f)

        logger.info(f"Predictions saved to {output_path}")

    except CustomException as e:
        logger.exception(f"Main pipeline failed: {e}")


# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    time_series_forecasts()
