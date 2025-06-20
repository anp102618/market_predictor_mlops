import yaml
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Generator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Model_Utils.regressor_models import ModelFactory
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml

# ------------------ Logger Setup ------------------ #
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml")

# ----------------- Config Paths ------------------ #
preprocessed_data_csv: Path = Path(config["Model_Tune_Evaluate"]["path"]["preprocessed_data_csv"])
tuned_model_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["tuned_model_yaml"])
regressors_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["regressors_yaml"])

# ---------------- Config Constants ---------------- #
allowed_models: List[str] = config["Model_Tune_Evaluate"]["const"]["allowed_models"]


class ModelTuner:
    """
    Class responsible for hyperparameter tuning and evaluation of regression models.
    """

    def __init__(self, config_path: Path, allowed_models: Optional[List[str]] = None):
        self.factory = ModelFactory(config_path)
        self.config_path = config_path
        self.allowed_models = allowed_models
        splitter = ScalingWithSplitStrategy()

        self.df = pd.read_csv(preprocessed_data_csv, index_col=[0])
        self.df.dropna(inplace=True)
        if 'date' in self.df.columns:
            self.df.drop(columns=['date'], inplace=True)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = splitter.apply(self.df)
        self.results = []

    def adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, p: int) -> Optional[float]:
        """Calculate adjusted R^2."""
        try:
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        except CustomException as e:
            logger.error(f"Error calculating adjusted R2: {e}")
            return None

    def generate_param_grid(self, param_dict: Dict[str, List[Any]]) -> Generator[Dict[str, Any], None, None]:
        """Yield parameter combinations from a dictionary of parameter lists."""
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    def evaluate_model(self, model_name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Evaluate model and return RMSE, MAE, R2, Adjusted R2."""
        try:
            preds = self.factory.predict(model, X, model_name)
            rmse = np.sqrt(mean_squared_error(y, preds))
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            adj_r2 = self.adjusted_r2(y, preds, X.shape[1])
            return rmse, mae, r2, adj_r2
        except CustomException as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return None, None, None, None

    @track_performance
    def run_grid_search(self) -> None:
        """Perform grid search across allowed models and evaluate best model."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except CustomException as e:
            logger.error(f"Failed to load config file: {e}")
            return

        for model_name, model_info in config.items():
            if self.allowed_models and model_name not in self.allowed_models:
                logger.info(f"Skipping {model_name} as it's not in the allowed list.")
                continue

            logger.info(f"Grid searching: {model_name}")
            param_grid = list(self.generate_param_grid(model_info['params']))
            best_r2_val = -np.inf
            best_model = None
            best_params = None

            for params in param_grid:
                try:
                    model = self.factory.get_model(model_name, param_override=params)
                    self.factory.train(model, self.X_train, self.y_train, model_name, self.X_val, self.y_val)
                    _, _, r2_val, _ = self.evaluate_model(model_name, model, self.X_val, self.y_val)

                    if r2_val is not None and r2_val > best_r2_val:
                        best_r2_val = r2_val
                        best_model = model
                        best_params = params
                except CustomException as e:
                    logger.warning(f"Skipping config {params} due to error: {e}")
                    continue

            if not best_params:
                logger.error(f"No valid hyperparameters found for {model_name}, skipping fine-tuning.")
                continue

            # Fine-tune on train + val
            logger.info(f"Best params for {model_name}: {best_params}")
            try:
                X_combined = np.concatenate([self.X_train, self.X_val])
                y_combined = np.concatenate([self.y_train, self.y_val])
                final_model = self.factory.get_model(model_name, best_params)
                self.factory.train(final_model, X_combined, y_combined, model_name)
            except CustomException as e:
                logger.error(f"Error during fine-tuning for {model_name}: {e}")
                continue

            # Final evaluation
            try:
                scores = {
                    "train": self.evaluate_model(model_name, final_model, self.X_train, self.y_train),
                    "val": self.evaluate_model(model_name, final_model, self.X_val, self.y_val),
                    "test": self.evaluate_model(model_name, final_model, self.X_test, self.y_test)
                }
                self.results.append({
                    "model": model_name,
                    "final_model": final_model,
                    "best_params": best_params,
                    "scores": {
                        split: dict(zip(["rmse", "mae", "r2", "adjusted_r2"], values))
                        for split, values in scores.items()
                    }
                })
            except CustomException as e:
                logger.error(f"Error during Evaluation for {model_name}: {e}")

    @track_performance
    def save_results(self, output_path: Path) -> None:
        """Save best model results to YAML."""
        try:
            sorted_results = sorted(self.results, key=lambda x: x['scores']['test']['adjusted_r2'] or -np.inf, reverse=True)
            cleaned_results = {}

            for item in sorted_results:
                model_class_name = item["final_model"].__class__.__name__
                cleaned_results[model_class_name] = {
                    "model": model_class_name,
                    "params": item["best_params"],
                    "scores": {
                        split: {k: round(float(v), 4) if v is not None else None for k, v in score.items()}
                        for split, score in item["scores"].items()
                    }
                }

            best = sorted_results[0]
            logger.info(
                f"Best model: {best['final_model'].__class__.__name__} with params: {best['best_params']} "
                f"and test Adjusted R²: {round(best['scores']['test']['adjusted_r2'], 4)}"
            )

            with open(output_path, 'w') as f:
                yaml.dump(cleaned_results, f, default_flow_style=False, sort_keys=False)

            print(f"\nResults saved to {output_path}")

        except CustomException as e:
            logger.error(f"Error during saving tuned models: {e}")


def execute_model_tune_evaluate():
    """Execute the full model tuning and evaluation pipeline."""
    try:
        logger.info("Starting Model Selection and Tuning...")
        tuner = ModelTuner(config_path=regressors_yaml, allowed_models=allowed_models)
        tuner.run_grid_search()
        tuner.save_results(tuned_model_yaml)
        logger.info("Selected model tuning completed successfully.")
    except CustomException as e:
        logger.error(f"Unexpected error in main(): {e}")


if __name__ == "__main__":
    execute_model_tune_evaluate()
