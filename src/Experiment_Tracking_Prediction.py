import os
import yaml
import json
import time
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy
from Common_Utils import (
    setup_logger, track_performance, CustomException,
    load_yaml, write_yaml, append_yaml, delete_joblib_model, copy_yaml_file
)
from Model_Utils.time_series_models import time_series_forecasts, add_average_to_yaml

# ------------------ Setup ------------------

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = "https://dagshub.com/anp102618/llm_nlp_tasks_mlops.mlflow"

logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml")

# ------------------ Config Paths ------------------
preprocessed_data_csv: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["preprocessed_data_csv"])
final_data_csv: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["final_data_csv"])
tuned_model_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["tuned_model_yaml"])
time_series_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["time_series_yaml"])
mlflow_details_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["mlflow_details_yaml"])
joblib_model_dir: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["joblib_model_dir"])
STAGE: str = config["Experiment_Tracking_Prediction"]["const"]["mlflow_stage"]

def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Compute adjusted R-squared."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, and Adjusted R2 metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "adjusted_r2": float(adjusted_r2(y_true, y_pred, n_features))
    }

def safe_log_metrics(metrics: Dict[str, float], prefix: str) -> None:
    """Log metrics to MLflow, catching any errors silently."""
    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}_{k}", float(v))
        except CustomException as e:
            logger.warning(f"Failed to log metric {prefix}_{k}: {e}")

@track_performance
def execute_mlflow_steps() -> None:
    """
    Main MLflow experiment tracking and prediction execution function.
    Includes model selection, training, evaluation, logging, and metadata saving.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        logger.info(f"MLflow tracking URI set to:{MLFLOW_TRACKING_URI}")

        config_model = load_yaml(tuned_model_yaml)
        first_key = next(iter(config_model))
        model_name = config_model[first_key]["model"]
        params = config_model[first_key]["params"]

        df = pd.read_csv(preprocessed_data_csv, index_col=0).drop(columns=["date"], errors="ignore")
        splitter = ScalingWithSplitStrategy()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.apply(df)

        model_cls = {"lasso": Lasso, "ridge": Ridge, "xgboost": XGBRegressor}.get(model_name.lower())
        if not model_cls:
            raise CustomException(f"Unsupported model: {model_name}")

        model = model_cls(**params)
        model.fit(X_train, y_train)

        train_metrics = evaluate_metrics(y_train, model.predict(X_train), X_train.shape[1])
        val_metrics = evaluate_metrics(y_val, model.predict(X_val), X_val.shape[1])

        experiment_name = "market-predictor-ml"
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")

            mlflow.log_param("model", model_name)
            mlflow.log_params(params)
            safe_log_metrics(train_metrics, "train")
            safe_log_metrics(val_metrics, "val")

            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))
            model.fit(X_combined, y_combined)

            test_metrics = evaluate_metrics(y_test[:-1], model.predict(X_test.iloc[:-1]), X_test.shape[1])
            safe_log_metrics(test_metrics, "test")

            last_row = X_test.iloc[[-1]]
            last_row_pred = float(model.predict(last_row)[0])
            mlflow.log_metric("last_xtest_row_prediction", last_row_pred)

            time_series_forecasts()
            append_yaml(time_series_yaml, {model_name: last_row_pred})
            df_final = pd.read_csv(final_data_csv, index_col=0)
            append_yaml(time_series_yaml, {"last_value": df_final['nsei'].iloc[-1]})
            add_average_to_yaml(time_series_yaml)

            model_path = Path("Tuned_Model/model.joblib")
            model_path_new = Path("Data/new_data/model.joblib")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            delete_joblib_model(model_path.parent)
            joblib.dump(model, model_path)
            joblib.dump(model, model_path_new)

            model_uri = f"runs:/{run_id}/model"
            metadata = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
                "model": model_name,
                "best_params": params,
                "scores": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
                "mlflow_run": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "model_uri": model_uri,
                    "artifact_uri": run.info.artifact_uri,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "last_row_prediction": last_row_pred,
                    "tracking_uri": MLFLOW_TRACKING_URI,
                },
                "saved_model_path": str(model_path.resolve())
            }

            with open(mlflow_details_yaml, "w") as f:
                yaml.dump(metadata, f)

            copy_yaml_file("Tuned_Model/mlflow_details.yaml", "Data/new_data")
            copy_yaml_file("Tuned_Model/time_series_predictions.yaml", "Data/new_data")

    except CustomException as ce:
        logger.exception(f"Custom error: {ce}")
    except CustomException as e:
        logger.exception(f"Unhandled exception: {e}")

if __name__ == "__main__":
    execute_mlflow_steps()
