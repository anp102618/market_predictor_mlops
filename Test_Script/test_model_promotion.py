import unittest
import yaml
import logging
import os
import pandas as pd
from scipy.stats import ks_2samp
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from Common_Utils import setup_logger, track_performance, CustomException

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
            with open(prev_metrics_path, 'r') as f:
                prev_data = yaml.safe_load(f)
            with open(curr_metrics_path, 'r') as f:
                curr_data = yaml.safe_load(f)

            prev_adj_r2 = prev_data.get("scores", {}).get("test", {}).get("adjusted_r2", None)
            curr_adj_r2 = curr_data.get("scores", {}).get("test", {}).get("adjusted_r2", None)

            if prev_adj_r2 is None or curr_adj_r2 is None:
                logger.warning("Adjusted R2 not found in one of the YAMLs. Skipping model drift check.")
                return False

            drop = prev_adj_r2 - curr_adj_r2
            logger.info(f"Previous adj_r2: {prev_adj_r2}, Current adj_r2: {curr_adj_r2}, Drop: {drop}")
            return drop > threshold

        except Exception as e:
            logger.error(f"Error in model drift detection: {e}")
            return False

    @track_performance
    def get_stage_from_run_id(self, client, model_name, run_id):
        try:
            model_versions = client.get_latest_versions(model_name)
            for version in model_versions:
                if version.run_id == run_id:
                    logger.info(f"Found version {version.version} for run_id {run_id} with stage {version.current_stage}")
                    return version.current_stage
            logger.warning(f"No model version found for run_id: {run_id}")
            return None
        except (MlflowException, CustomException) as e:
            logger.error(f"Error while fetching model stage: {e}")
            raise

    @track_performance
    def get_model_version(self, client, model_name, run_id):
        model_versions = client.get_latest_versions(model_name)
        for version in model_versions:
            if version.run_id == run_id:
                return version.version
        raise ValueError(f"No model version found for run_id: {run_id}")

    @track_performance
    def test_model_promotion(self):
        try:
            yaml_path = "Tuned_Model/mlflow_details.yaml"
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            client = MlflowClient()
            model_name = data.get("model", None)
            run_id = data.get("mlflow_run", {}).get("run_id", None)

            self.assertIsNotNone(model_name, "Model name missing from YAML.")
            self.assertIsNotNone(run_id, "Run ID missing from YAML.")

            current_stage = self.get_stage_from_run_id(client, model_name, run_id)
            self.assertIsNotNone(current_stage, f"Could not determine stage for run_id: {run_id}")

            # === Score Check
            scores = data.get("scores", {})
            adj_r2s = []
            for dataset in ["train", "val", "test"]:
                r2 = scores.get(dataset, {}).get("r2", 0)
                adj_r2 = scores.get(dataset, {}).get("adjusted_r2", 0)
                logger.info(f"{dataset} R2: {r2}, Adjusted R2: {adj_r2}")
                self.assertGreaterEqual(r2, 0.9, f"{dataset} r2 should be >= 0.9")
                self.assertGreaterEqual(adj_r2, 0.9, f"{dataset} adjusted_r2 should be >= 0.9")
                adj_r2s.append(adj_r2)

            high_adj_r2 = all(score >= 0.9 for score in adj_r2s)

            # === Drift Checks
            data_drift = self.check_data_drift(
                "Data/previous_data/final_data.csv",
                "Data/processed_data/final_data.csv"
            )

            model_drift = self.check_model_drift(
                "Data/previous_data/mlflow_details.yaml",
                "Tuned_Model/mlflow_details.yaml"
            )

            # === Load Predictions
            with open("Tuned_Model/time_series_predictions.yaml", "r") as f:
                preds_data = yaml.safe_load(f)
            lasso_val = float(preds_data.get("Lasso", -1))
            avg_val = float(preds_data.get("average_expected", -1))
            last_val = float(preds_data.get("last_value", -1))
            logger.info(f"Lasso: {lasso_val}, Avg: {avg_val}, Last: {last_val}")

            prediction_logic = (lasso_val > last_val and avg_val > last_val) or \
                               (lasso_val < last_val and avg_val < last_val)

            # === Promotion Decision
            promotion_allowed = high_adj_r2 and not data_drift and not model_drift and prediction_logic

            if promotion_allowed:
                if current_stage.lower() == "staging":
                    version = self.get_model_version(client, model_name, run_id)
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    logger.info(f"Promoted model version {version} to Production.")

                    # Update YAML
                    if "mlflow_run" not in data:
                        data["mlflow_run"] = {}
                    data["mlflow_run"]["current_stage"] = "Production"

                    with open(yaml_path, "w") as f:
                        yaml.dump(data, f)
                    logger.info("Updated mlflow_details.yaml with stage = Production")

                    #  Write promotion success flag
                    os.makedirs("Test_Script", exist_ok=True)
                    with open("Test_Script/promotion_success.txt", "w") as f:
                        f.write("promoted")

                    #  Final assertion
                    promoted_stage = self.get_stage_from_run_id(client, model_name, run_id)
                    self.assertEqual(promoted_stage.lower(), "production", "Model was not promoted to Production")
                else:
                    logger.info(f"Model already in stage {current_stage}. No promotion required.")
            else:
                logger.info("Promotion conditions not met – no promotion.")
                self.assertTrue(True, "Promotion conditions not satisfied – skipping promotion.")

        except Exception as e:
            logger.error(f"Unexpected error in test model promotion: {e}")
            self.fail(f"Exception occurred: {e}")

def execute_test_script():
    try:
        logger.info("Starting the Test script run ....")
        unittest.main()
        logger.info("Test script run completed successfully..")
    except Exception as e:
        logger.error(f"Unexpected error in test script run : {e}")

if __name__ == "__main__":
    unittest.main()  