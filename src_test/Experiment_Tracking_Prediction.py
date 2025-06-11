import pandas as pd
import numpy as np
import yaml
import mlflow
import os
import mlflow.sklearn
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml, write_yaml, append_yaml,delete_joblib_model
from Model_Utils.time_series_models import time_series_forecasts, add_average_to_yaml


# ------------------ Logger Setup ------------------ #
# Setup logger
logger = setup_logger(filename="logs")

# ------------------ Setup ------------------ 
TUNED_MODELS_YAML_PATH = "Config_Yaml/tuned_model.yaml"
STAGE = "Staging"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# ------------------ Utility Functions ------------------ #
def adjusted_r2(y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    first_key = next(iter(config))
    return config[first_key]

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
            value = float(v.item() if isinstance(v, np.generic) else v)
            mlflow.log_metric(f"{prefix}_{k}", value)
        except Exception as e:
            logger.warning(f"Failed to log metric {prefix}_{k}: {e} (value: {v})")

# ------------------ Main Pipeline ------------------ #
def run_pipeline():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        date = datetime.now().strftime("%Y%m%d")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")

        config = load_config(TUNED_MODELS_YAML_PATH)
        model_name = config["model"]
        params = config["params"]
        
        splitter = ScalingWithSplitStrategy()
        df = pd.read_csv("Data/processed_data/preprocessed_data.csv", index_col=[0])
        df = df.drop(columns=['date'])
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.apply(df)
        
        if model_name.lower() == "lasso":
            model = Lasso(**params)
        elif model_name.lower() == "ridge":
            model = Ridge(**params)

        elif model_name.lower() == "xgboost":
            model = XGBRegressor(**params)
        else:
            raise CustomException(f"Unsupported model: {model_name}")
        
        model.fit(X_train, y_train)
        logger.info(f"Model {model_name} trained on training set.")

        train_metrics = evaluate_metrics(y_train, model.predict(X_train), X_train.shape[1])
        val_metrics = evaluate_metrics(y_val, model.predict(X_val), X_val.shape[1])

        experiment_name = f"Experiment_Lasso_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = f"{model_name}_{timestamp}"
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name=run_name) as run:
            run_id_value = run.info.run_id
            logger.info(f"Started MLflow run: {run_id_value}")

            mlflow.log_param("model", model_name)
            mlflow.log_params(params)

            safe_log_metrics(train_metrics, "train")
            safe_log_metrics(val_metrics, "val")

            # Fine-tune on train + val
            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))
            model.fit(X_combined, y_combined)
            test_metrics = evaluate_metrics(y_test[:-1], model.predict(X_test.iloc[:-1]), X_test.shape[1])
            safe_log_metrics(test_metrics, "test")

            #Time Series Predictions 
            time_series_forecasts()
            
            #last row prediction for next day
            last_row = X_test.iloc[[-1]]
            last_row_pred = float(model.predict(last_row)[0])
            mlflow.log_metric("last_xtest_row_prediction", last_row_pred)
            append_yaml("Tuned_Model/time_series_predictions.yaml",{model_name:last_row_pred})
            logger.info(f"Prediction for the last row of X_test: {last_row_pred}")

            df = pd.read_csv("Data/processed_data/final_data.csv", index_col=[0])
            append_yaml("Tuned_Model/time_series_predictions.yaml",{"last_value":df['nsei'].iloc[-1]})

            add_average_to_yaml("Tuned_Model/time_series_predictions.yaml")

            model_filename = f"{model_name}_model_{timestamp}.joblib"
            
            #root_dir = Path(os.getcwd())
            root_dir = Path(__file__).resolve().parent.parent
            model_dir = root_dir / "Tuned_Model"
            model_dir.mkdir(parents=True, exist_ok=True)
            #model_path = model_dir / model_filename
            model_path = root_dir / "Tuned_Model" / model_filename
            relative_model_path = model_path.relative_to(root_dir)
            delete_joblib_model("Tuned_Model/")
            joblib.dump(model, relative_model_path)
            logger.info(f"Model saved locally to: {model_path}")

            mlflow.sklearn.log_model(model, artifact_path=model_name)
            logger.info("Model artifact logged to MLflow.")

            client = MlflowClient()
            model_uri = f"runs:/{run_id_value}/artifacts/{model_name}"
            reg = mlflow.register_model(model_uri, model_name)
            model_version = reg.version
            logger.info(f"Model registered: version {model_version}")

            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=STAGE,
                archive_existing_versions=True
            )

            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="version_status",
                value=STAGE
            )

            logger.info(f"Model transitioned to {STAGE} stage and tagged.")

            mlflow_details = {
                "timestamp": timestamp,
                "model": model_name,
                "best_params": params,
                "scores": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics
                },
                "mlflow_run": {
                    "run_id": run_id_value,
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

            model_dir_mlflow = Path("Tuned_Model")
            model_filename_mlflow = "mlflow_details.yaml"
            model_dir_mlflow.mkdir(parents=True, exist_ok=True)
            model_path_mlflow = model_dir / model_filename_mlflow
            with open(model_path_mlflow, "w") as f:
                yaml.dump(mlflow_details, f)

            logger.info(f"All model details saved to: {model_path_mlflow}")

    except CustomException as e:
        logger.exception("Pipeline failed due to an error.: {e}")
        

# ------------------ Entry Point ------------------ #
if __name__ == "__main__":
    run_pipeline()
