import os
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy
from Common_Utils import (
    setup_logger, track_performance, CustomException,
    load_yaml, write_yaml, append_yaml, delete_joblib_model
)
from Model_Utils.time_series_models import time_series_forecasts, add_average_to_yaml
from dotenv import load_dotenv

# ------------------ Load Environment ------------------ 

# Optional: load from .env if running locally
load_dotenv()

# Load from environment (GitHub Actions or local .env)
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize Dagshub integration with token
dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name="market_predictor_mlops",
    mlflow=True,
    token=DAGSHUB_TOKEN
)
logger = setup_logger(filename="logs")

# ------------------ Load Config ------------------ #
config = load_yaml("Config_Yaml/model_config.yaml")
paths = config["Experiment_Tracking_Prediction"]["path"]

preprocessed_data_csv = Path(paths["preprocessed_data_csv"])
final_data_csv = Path(paths["final_data_csv"])
tuned_model_yaml = Path(paths["tuned_model_yaml"])
time_series_yaml = Path(paths["time_series_yaml"])
mlflow_details_yaml = Path(paths["mlflow_details_yaml"])
joblib_model_dir = Path(paths["joblib_model_dir"])

STAGE = config["Experiment_Tracking_Prediction"]["const"]["mlflow_stage"]

# ------------------ Helpers ------------------ #
def adjusted_r2(y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def evaluate_metrics(y_true, y_pred, n_features):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "adjusted_r2": float(adjusted_r2(y_true, y_pred, n_features))
    }

def safe_log_metrics(metrics: dict, prefix: str):
    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}_{k}", float(v))
        except Exception as e:
            logger.warning(f"Failed to log metric {prefix}_{k}: {e}")

# ------------------ Main MLflow Pipeline ------------------ #
@track_performance
def execute_mlflow_steps():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        date = datetime.now().strftime("%Y%m%d")

        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

        config_model = load_yaml(tuned_model_yaml)
        model_name = config_model["model"]
        params = config_model["params"]

        df = pd.read_csv(preprocessed_data_csv, index_col=0).drop(columns=["date"])
        splitter = ScalingWithSplitStrategy()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.apply(df)

        model_cls = {
            "lasso": Lasso,
            "ridge": Ridge,
            "xgboost": XGBRegressor
        }.get(model_name.lower())

        if model_cls is None:
            raise CustomException(f"Unsupported model: {model_name}")

        model = model_cls(**params)
        model.fit(X_train, y_train)

        train_metrics = evaluate_metrics(y_train, model.predict(X_train), X_train.shape[1])
        val_metrics = evaluate_metrics(y_val, model.predict(X_val), X_val.shape[1])

        experiment_name = f"Experiment_{model_name}_{timestamp}"
        run_name = f"{model_name}_{timestamp}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")

            mlflow.log_param("model", model_name)
            mlflow.log_params(params)
            safe_log_metrics(train_metrics, "train")
            safe_log_metrics(val_metrics, "val")

            # Combine train+val for final training
            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))
            model.fit(X_combined, y_combined)

            test_metrics = evaluate_metrics(y_test[:-1], model.predict(X_test.iloc[:-1]), X_test.shape[1])
            safe_log_metrics(test_metrics, "test")

            # Predict final row
            last_row = X_test.iloc[[-1]]
            last_row_pred = float(model.predict(last_row)[0])
            mlflow.log_metric("last_xtest_row_prediction", last_row_pred)

            # Time series logging
            time_series_forecasts()
            append_yaml(time_series_yaml, {model_name: last_row_pred})
            df_final = pd.read_csv(final_data_csv, index_col=0)
            append_yaml(time_series_yaml, {"last_value": df_final['nsei'].iloc[-1]})
            add_average_to_yaml(time_series_yaml)

            # Save & log model
            model_path = Path("Tuned_Model/model.joblib")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            delete_joblib_model(model_path.parent)
            joblib.dump(model, model_path)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, artifact_path=model_name, input_example=X_train.iloc[:1], signature=signature)
            logger.info("Model artifact logged to MLflow.")

            # Register and transition
            model_uri = f"runs:/{run_id}/{model_name}"
            registered = mlflow.register_model(model_uri, model_name)
            model_version = registered.version

            client = MlflowClient()
            client.transition_model_version_stage(model_name, model_version, stage=STAGE, archive_existing_versions=True)
            client.set_model_version_tag(model_name, model_version, "version_status", STAGE)

            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "model": model_name,
                "best_params": params,
                "scores": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
                "mlflow_run": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "model_uri": model_uri,
                    "model_version": model_version,
                    "current_stage": STAGE,
                    "artifact_uri": run.info.artifact_uri,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "last_row_prediction": last_row_pred,
                    "tracking_uri": MLFLOW_TRACKING_URI
                },
                "saved_model_path": str(model_path.resolve())
            }

            with open(mlflow_details_yaml, "w") as f:
                yaml.dump(metadata, f)
            logger.info(f"Saved MLflow metadata to: {mlflow_details_yaml}")

    except CustomException as ce:
        logger.exception(f"Custom error: {ce}")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")

if __name__ == "__main__":
    execute_mlflow_steps()
