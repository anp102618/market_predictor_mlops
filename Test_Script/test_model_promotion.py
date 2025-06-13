import os
import yaml
import unittest
import logging
import pandas as pd
from scipy.stats import ks_2samp
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


class SimpleModelTest(unittest.TestCase):

    def setUp(self):
        self.curr_yaml_path = "Tuned_Model/mlflow_details.yaml"
        self.pred_yaml_path = "Tuned_Model/time_series_predictions.yaml"
        self.drift = DriftChecker(
            "Data/previous_data/final_data.csv",
            "Data/processed_data/final_data.csv",
            "Data/previous_data/mlflow_details.yaml",
            self.curr_yaml_path
        )

    def adj_r2_score(self):
        scores = load_yaml(self.curr_yaml_path).get("scores", {})
        for split in ["train", "val", "test"]:
            r2 = scores.get(split, {}).get("r2", 0)
            adj_r2 = scores.get(split, {}).get("adjusted_r2", 0)
            if r2 < 0.9 or adj_r2 < 0.9:
                return False
        return True

    def time_series_pred(self):
        preds = load_yaml(self.pred_yaml_path)
        lasso = preds.get("Lasso", -1)
        avg = preds.get("average_expected", -1)
        last = preds.get("last_value", -1)
        return (lasso > last and avg > last) or (lasso < last and avg < last)

    @track_performance
    def test_model_passes(self):
        reasons = []

        try:
            if not self.adj_r2_score():
                reasons.append("Low model score")

            if self.drift.data_drift():
                reasons.append("Data drift detected")

            if self.drift.model_drift():
                reasons.append("Model drift detected")

            if not self.time_series_pred():
                reasons.append("Prediction jump inconsistent")

            if not reasons:
                # Update YAML to reflect promotion
                with open(self.curr_yaml_path, "r") as f:
                    meta = yaml.safe_load(f)

                meta["mlflow_run"]["stage"] = "production"

                with open(self.curr_yaml_path, "w") as f:
                    yaml.safe_dump(meta, f)

                print("Test Passed")
            else:
                print("Test Failed:", ", ".join(reasons))

        except Exception as e:
            logger.error(f"Unhandled test error: {e}")
            print("Test Failed: Unexpected error")


if __name__ == "__main__":
    unittest.main()
