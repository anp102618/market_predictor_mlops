import os
import json
import yaml
import unittest
import logging
import pandas as pd
from scipy.stats import ks_2samp
from mlflow.tracking import MlflowClient
from Common_Utils import setup_logger, load_yaml, track_performance

logger = setup_logger("logs")


class DriftChecker:
    def __init__(self, ref_data, new_data, ref_metrics, new_metrics):
        self.ref_data = pd.read_csv(ref_data)
        self.new_data = pd.read_csv(new_data)
        self.prev_metrics = load_yaml(ref_metrics)
        self.curr_metrics = load_yaml(new_metrics)

    def data_drift(self, threshold=0.05):
        for col in self.ref_data.columns:
            if col in self.new_data.columns:
                stat, p_value = ks_2samp(self.ref_data[col].dropna(), self.new_data[col].dropna())
                logger.info(f"Drift p-value for {col}: {p_value:.4f}")
                if p_value < threshold:
                    logger.warning(f"Drift detected in column: {col}")
                    return True
        return False

    def model_drift(self, threshold=0.05):
        prev = self.prev_metrics.get("scores", {}).get("test", {}).get("adjusted_r2")
        curr = self.curr_metrics.get("scores", {}).get("test", {}).get("adjusted_r2")

        if prev is None or curr is None:
            logger.warning("Missing adjusted R² values.")
            return False

        drop = prev - curr
        logger.info(f"Model drift: prev_adj={prev}, curr_adj={curr}, drop={drop:.4f}")
        return drop > threshold


class ModelPromotionTest(unittest.TestCase):

    def setUp(self):
        self.client = MlflowClient()
        self.experiment_name = "market-predictor-ml"
        self.model_name, self.run_id = self._get_latest_model_info()
        self.drift = DriftChecker(
            "Data/previous_data/final_data.csv",
            "Data/processed_data/final_data.csv",
            "Data/previous_data/mlflow_details.yaml",
            "Tuned_Model/mlflow_details.yaml"
        )

    def _get_latest_model_info(self):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=5)

        for run in runs:
            tag = run.data.tags.get("mlflow.log-model.history")
            if tag:
                try:
                    model_hist = json.loads(tag)
                    return model_hist[0]["name"], run.info.run_id
                except Exception as e:
                    logger.warning(f"Tag parsing failed: {e}")

        # Fallback
        config = load_yaml("Config_Yaml/model_config.yaml")
        tuned_path = config["Experiment_Tracking_Prediction"]["path"]["tuned_model_yaml"]
        model_yaml = load_yaml(tuned_path)
        return next(iter(model_yaml)), runs[0].info.run_id

    def _get_stage(self):
        for version in self.client.get_latest_versions(self.model_name):
            if version.run_id == self.run_id:
                return version.current_stage
        return None

    def _get_model_version(self):
        for version in self.client.get_latest_versions(self.model_name):
            if version.run_id == self.run_id:
                return version.version
        raise ValueError("Version not found.")

    def _high_score(self):
        scores = load_yaml("Tuned_Model/mlflow_details.yaml").get("scores", {})
        for split in ["train", "val", "test"]:
            r2 = scores.get(split, {}).get("r2", 0)
            adj_r2 = scores.get(split, {}).get("adjusted_r2", 0)
            logger.info(f"{split.upper()} R²: {r2}, Adj R²: {adj_r2}")
            if r2 < 0.9 or adj_r2 < 0.9:
                return False
        return True

    def _prediction_jump_ok(self):
        preds = load_yaml("Tuned_Model/time_series_predictions.yaml")
        lasso = preds.get("Lasso", -1)
        avg = preds.get("average_expected", -1)
        last = preds.get("last_value", -1)
        logger.info(f"Lasso={lasso}, Avg={avg}, Last={last}")
        return (lasso > last and avg > last) or (lasso < last and avg < last)

    @track_performance
    def test_model_promotion(self):
        stage = self._get_stage()
        self.assertIsNotNone(stage, "Stage not found.")

        if all([
            self._high_score(),
            not self.drift.data_drift(),
            not self.drift.model_drift(),
            self._prediction_jump_ok()
        ]):

            if stage.lower() == "staging":
                version = self._get_model_version()
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f" Promoted model '{self.model_name}' v{version} to PRODUCTION")

                with open("Test_Script/promotion_success.txt", "w") as f:
                    f.write("promoted")

                self.assertEqual(
                    self._get_stage().lower(), "production", "Model was not promoted."
                )
            else:
                logger.info(f"Model already in stage '{stage}', no need to promote.")

        else:
            logger.info(" Conditions not met. Model not promoted.")
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
