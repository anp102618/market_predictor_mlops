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
from mlflow.models import infer_signature


# ------------------ Logger Setup ------------------ #
# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml") 

#-------------------config-paths--------------#
preprocessed_data_csv: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["preprocessed_data_csv"])
final_data_csv: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["final_data_csv"])
tuned_model_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["tuned_model_yaml"])
time_series_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["time_series_yaml"])
mlflow_details_yaml: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["mlflow_details_yaml"])
joblib_model_dir: Path = Path(config["Experiment_Tracking_Prediction"]["path"]["joblib_model_dir"])

#------------------config-const--------------------------#
MLFLOW_TRACKING_URI: str = config["Experiment_Tracking_Prediction"]["const"]["mlflow_tracking_uri"]
STAGE: str = config["Experiment_Tracking_Prediction"]["const"]["mlflow_stage"]



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

def execute_mlflow_steps():
    try:
        # ----- Setup -----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        date = datetime.now().strftime("%Y%m%d")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")

        # ----- Load config/model -----
        config_model = load_config(tuned_model_yaml)
        model_name = config_model["model"]
        params = config_model["params"]

        # ----- Load and split data -----
        df = pd.read_csv(preprocessed_data_csv, index_col=[0]).drop(columns=['date'])
        splitter = ScalingWithSplitStrategy()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.apply(df)

        # ----- Initialize model -----
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

        # ----- Evaluate metrics -----
        train_metrics = evaluate_metrics(y_train, model.predict(X_train), X_train.shape[1])
        val_metrics = evaluate_metrics(y_val, model.predict(X_val), X_val.shape[1])

        # ----- MLflow run -----
        experiment_name = f"Experiment_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = f"{model_name}_{timestamp}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            run_id_value = run.info.run_id
            logger.info(f"Started MLflow run: {run_id_value}")

            mlflow.log_param("model", model_name)
            mlflow.log_params(params)
            safe_log_metrics(train_metrics, "train")
            safe_log_metrics(val_metrics, "val")

            # ----- Refit and evaluate on test -----
            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))
            model.fit(X_combined, y_combined)
            test_metrics = evaluate_metrics(y_test[:-1], model.predict(X_test.iloc[:-1]), X_test.shape[1])
            safe_log_metrics(test_metrics, "test")

            # ----- Time Series Predictions -----
            time_series_forecasts()

            # ----- Predict next day -----
            last_row = X_test.iloc[[-1]]
            last_row_pred = float(model.predict(last_row)[0])
            mlflow.log_metric("last_xtest_row_prediction", last_row_pred)
            append_yaml(time_series_yaml, {model_name: last_row_pred})
            logger.info(f"Prediction for the last row of X_test: {last_row_pred}")

            df_final = pd.read_csv(final_data_csv, index_col=[0])
            append_yaml(time_series_yaml, {"last_value": df_final['nsei'].iloc[-1]})
            add_average_to_yaml(time_series_yaml)

            # ----- Save locally (joblib) -----
            root_dir = Path(__file__).resolve().parent.parent
            model_path = root_dir / "Tuned_Model" / "model.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            delete_joblib_model(model_path.parent)
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally to: {model_path}")

            # ----- MLflow model logging -----
            input_example = X_train.iloc[:1]
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, artifact_path=model_name,
                input_example=input_example, signature=signature
            )
            logger.info("Model artifact logged to MLflow.")

            # ----- Register model -----
            model_uri = f"runs:/{run_id_value}/{model_name}"
            reg = mlflow.register_model(model_uri, model_name)
            model_version = reg.version
            logger.info(f"Model registered: version {model_version}")

            # ----- Transition to STAGE -----
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=STAGE,
                archive_existing_versions=True
            )
            client.set_model_version_tag(
                name=model_name, version=model_version,
                key="version_status", value=STAGE
            )
            logger.info(f"Model transitioned to {STAGE} stage and tagged.")

            # ----- Save model metadata -----
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

            metadata_path = model_path.parent / "mlflow_details.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(mlflow_details, f)
            logger.info(f"All model details saved to: {metadata_path}")

    except CustomException as e:
        logger.exception(f"Pipeline failed due to custom exception: {e}")
    except Exception as e:
        logger.exception(f"Unhandled error in pipeline: {e}")


# ------------------ Entry Point ------------------ #
if __name__ == "__main__":
    execute_mlflow_steps()
