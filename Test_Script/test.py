import unittest
import yaml
import logging
import os
import json
import pandas as pd
from scipy.stats import ks_2samp
from mlflow.tracking import MlflowClient
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml

logger = setup_logger(filename="logs")


class TestModelPromotion(unittest.TestCase):

    def check_data_drift(self, ref_path, new_path, threshold=0.05):
        ref = pd.read_csv(ref_path)
        new = pd.read_csv(new_path)
        drift_detected = False

        for column in ref.columns:
            if column in new.columns:
                stat, p_value = ks_2samp(ref[column].dropna(), new[column].dropna())
                logger.info(f"Data drift p-value for {column}: {p_value:.4f}")
                if p_value < threshold:
                    drift_detected = True
                    logger.warning(f"Data drift detected in column: {column}")
        return drift_detected

    def check_model_drift(self, prev_metrics_path, curr_metrics_path, threshold=0.05):
        try:
            prev = load_yaml(prev_metrics_path)
            curr = load_yaml(curr_metrics_path)

            prev_adj = prev.get("scores", {}).get("test", {}).get("adjusted_r2")
            curr_adj = curr.get("scores", {}).get("test", {}).get("adjusted_r2")

            if prev_adj is None or curr_adj is None:
                logger.warning("Missing adjusted R² in YAMLs – skipping model drift check.")
                return False

            drop = prev_adj - curr_adj
            logger.info(f"Model drift check: Prev adj_r2={prev_adj}, Curr adj_r2={curr_adj}, Drop={drop}")
            return drop > threshold

        except Exception as e:
            logger.error(f"Error checking model drift: {e}")
            return False

    def get_latest_model_info(self, client, experiment_name="market-predictor-ml"):
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=5)

        if not runs:
            raise RuntimeError("No runs found in the experiment.")

        for run in runs:
            tag = run.data.tags.get("mlflow.log-model.history")
            if tag:
                try:
                    model_hist = json.loads(tag)
                    name = model_hist[0]["name"]
                    return name, run.info.run_id
                except Exception as e:
                    logger.warning(f"Failed to parse tag from run {run.info.run_id}: {e}")

        # fallback from YAML
        try:
            config = load_yaml("Config_Yaml/model_config.yaml")
            tuned_path = config["Experiment_Tracking_Prediction"]["path"]["tuned_model_yaml"]
            tuned_model = load_yaml(tuned_path)
            model_name = next(iter(tuned_model))
            return model_name, runs[0].info.run_id
        except Exception as e:
            raise RuntimeError(f"Fallback failed: {e}")

    @track_performance
    def get_stage_from_run_id(self, client, model_name, run_id):
        for version in client.get_latest_versions(model_name):
            if version.run_id == run_id:
                logger.info(f"Version {version.version} of model {model_name} is in stage {version.current_stage}")
                return version.current_stage
        return None

    @track_performance
    def get_model_version(self, client, model_name, run_id):
        for version in client.get_latest_versions(model_name):
            if version.run_id == run_id:
                return version.version
        raise ValueError(f"No model version found for run_id: {run_id}")

    @track_performance
    def test_model_promotion(self):
        try:
            client = MlflowClient()
            model_name, run_id = self.get_latest_model_info(client)

            current_stage = self.get_stage_from_run_id(client, model_name, run_id)
            self.assertIsNotNone(current_stage, f"Stage not found for run_id {run_id}")

            scores = load_yaml("Tuned_Model/mlflow_details.yaml").get("scores", {})
            adj_r2s = []

            for split in ["train", "val", "test"]:
                r2 = scores.get(split, {}).get("r2", 0)
                adj_r2 = scores.get(split, {}).get("adjusted_r2", 0)
                logger.info(f"{split.capitalize()} R² = {r2}, Adjusted R² = {adj_r2}")
                self.assertGreaterEqual(r2, 0.9, f"{split} r2 < 0.9")
                self.assertGreaterEqual(adj_r2, 0.9, f"{split} adj_r2 < 0.9")
                adj_r2s.append(adj_r2)

            high_score = all(val >= 0.9 for val in adj_r2s)
            data_drift = self.check_data_drift("Data/previous_data/final_data.csv", "Data/processed_data/final_data.csv")
            model_drift = self.check_model_drift("Data/previous_data/mlflow_details.yaml", "Tuned_Model/mlflow_details.yaml")

            preds = load_yaml("Tuned_Model/time_series_predictions.yaml")
            lasso_val = float(preds.get("Lasso", -1))
            avg_val = float(preds.get("average_expected", -1))
            last_val = float(preds.get("last_value", -1))

            logger.info(f"Lasso = {lasso_val}, Avg = {avg_val}, Last = {last_val}")
            logical_jump = (lasso_val > last_val and avg_val > last_val) or (lasso_val < last_val and avg_val < last_val)

            if high_score and not data_drift and not model_drift and logical_jump:
                if current_stage.lower() == "staging":
                    version = self.get_model_version(client, model_name, run_id)
                    client.transition_model_version_stage(model_name, version, stage="Production", archive_existing_versions=True)
                    logger.info(f"Promoted {model_name} v{version} to Production")

                    os.makedirs("Test_Script", exist_ok=True)
                    with open("Test_Script/promotion_success.txt", "w") as f:
                        f.write("promoted")

                    new_stage = self.get_stage_from_run_id(client, model_name, run_id)
                    self.assertEqual(new_stage.lower(), "production", "Model was not promoted properly.")
                else:
                    logger.info(f"Model already in stage '{current_stage}', no promotion.")
            else:
                logger.info("Conditions not met. Model not promoted.")
                self.assertTrue(True)

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            self.fail(f"Test failed: {e}")


if __name__ == "__main__":
    unittest.main()
