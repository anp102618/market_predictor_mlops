import os
import yaml
import unittest
import logging
import pandas as pd
from typing import Optional
from scipy.stats import ks_2samp
from Common_Utils import setup_logger, load_yaml, track_performance, CustomException

# Setup logger
logger = setup_logger("logs")


class DriftChecker:
    """
    A utility class to check data and model drift using statistical tests and metric comparisons.
    """

    def __init__(self, ref_data: str, new_data: str, ref_metrics: str, new_metrics: str):
        """
        Initialize with file paths to reference and new datasets and metrics.

        Args:
            ref_data (str): Path to reference data CSV.
            new_data (str): Path to current data CSV.
            ref_metrics (str): Path to reference metrics YAML.
            new_metrics (str): Path to current metrics YAML.
        """
        self.ref_data = pd.read_csv(ref_data)
        self.new_data = pd.read_csv(new_data)
        self.prev_metrics = load_yaml(ref_metrics)
        self.curr_metrics = load_yaml(new_metrics)

    def data_drift(self, threshold: float = 0.05) -> bool:
        """
        Detects data drift using the KS test for numeric columns.

        Args:
            threshold (float): Significance level for drift detection.

        Returns:
            bool: True if any drift is detected, False otherwise.
        """
        for col in self.ref_data.columns:
            if col in self.new_data.columns:
                try:
                    stat, p_value = ks_2samp(self.ref_data[col].dropna(), self.new_data[col].dropna())
                    logger.info(f"Drift p-value for {col}: {p_value:.4f}")
                    if p_value < threshold:
                        logger.warning(f"Data drift detected in column: {col}")
                        return True
                except CustomException as e:
                    logger.warning(f"KS test failed for {col}: {e}")
        return False

    def model_drift(self, threshold: float = 0.05) -> bool:
        """
        Detects model drift using a drop in adjusted R².

        Args:
            threshold (float): Acceptable drop before drift is flagged.

        Returns:
            bool: True if model performance dropped significantly.
        """
        try:
            prev = self.prev_metrics.get("scores", {}).get("test", {}).get("adjusted_r2")
            curr = self.curr_metrics.get("scores", {}).get("test", {}).get("adjusted_r2")

            if prev is None or curr is None:
                logger.warning("Adjusted R² value missing in one of the YAML files.")
                return False

            drop = prev - curr
            logger.info(f"Model drift check - prev_adj: {prev}, curr_adj: {curr}, drop: {drop:.4f}")
            return drop > threshold

        except CustomException as e:
            logger.error(f"Error in model_drift: {e}")
            return False


class SimpleModelTest(unittest.TestCase):
    """
    Unit test for validating model and data integrity before deployment.
    """

    def setUp(self) -> None:
        """
        Setup test case by initializing file paths and drift checker.
        """
        self.curr_yaml_path = "Tuned_Model/mlflow_details.yaml"
        self.pred_yaml_path = "Tuned_Model/time_series_predictions.yaml"

        self.drift = DriftChecker(
            ref_data="Data/previous_data/final_data.csv",
            new_data="Data/processed_data/final_data.csv",
            ref_metrics="Data/previous_data/mlflow_details.yaml",
            new_metrics=self.curr_yaml_path
        )

    def adj_r2_score(self) -> bool:
        """
        Verifies that R² and adjusted R² are consistently high across splits.

        Returns:
            bool: True if all scores are above threshold.
        """
        try:
            scores = load_yaml(self.curr_yaml_path).get("scores", {})
            for split in ["train", "val", "test"]:
                r2 = scores.get(split, {}).get("r2", 0)
                adj_r2 = scores.get(split, {}).get("adjusted_r2", 0)
                logger.info(f"{split.upper()} R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")
                if r2 < 0.9 or adj_r2 < 0.9:
                    return False
            return True
        except CustomException as e:
            logger.error(f"Error in adj_r2_score check: {e}")
            return False

    def time_series_pred(self) -> bool:
        """
        Checks that predictions follow same direction as average and last values.

        Returns:
            bool: True if Lasso and Average predictions follow same trend as last.
        """
        try:
            preds = load_yaml(self.pred_yaml_path)
            lasso = preds.get("Lasso", -1)
            avg = preds.get("average_expected", -1)
            last = preds.get("last_value", -1)

            if lasso == -1 or avg == -1 or last == -1:
                logger.warning("One or more prediction keys are missing.")
                return False

            consistent_up = lasso > last and avg > last
            consistent_down = lasso < last and avg < last
            return consistent_up or consistent_down
        except CustomException as e:
            logger.error(f"Error in time_series_pred check: {e}")
            return False

    @track_performance
    def test_model_passes(self) -> None:
        """
        Runs all checks and prints outcome.
        Promotes model to production if it passes.
        """
        reasons = []

        try:
            if not self.adj_r2_score():
                reasons.append("Low model R²/Adjusted R² score")

            if self.drift.data_drift():
                reasons.append("Data drift detected")

            if self.drift.model_drift():
                reasons.append("Model drift detected")

            if not self.time_series_pred():
                reasons.append("Prediction direction inconsistent")

            if not reasons:
                logger.info("All checks passed. Promoting model.")
                with open(self.curr_yaml_path, "r") as f:
                    meta = yaml.safe_load(f)
                meta.setdefault("mlflow_run", {})["stage"] = "production"
                with open(self.curr_yaml_path, "w") as f:
                    yaml.safe_dump(meta, f)
                print("Test Passed")
            else:
                logger.warning(f"Test failed due to: {reasons}")
                print("Test Failed:", ", ".join(reasons))

        except CustomException as e:
            logger.exception(f"Unhandled error in test_model_passes: {e}")
            print("Test Failed: Unexpected error")


if __name__ == "__main__":
    unittest.main()
