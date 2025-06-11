import yaml
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from Model_Utils.regressor_models import ModelFactory 
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml


# ------------------ Logger Setup ------------------ #
# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml")

#-----------------Config-paths------------#
preprocessed_data_csv: Path = Path(config["Model_Tune_Evaluate"]["path"]["preprocessed_data_csv"])
tuned_model_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["tuned_model_yaml"])
regressors_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["regressors_yaml"])

# ----------------Config-const----------#
allowed_models: list[str] = config["Model_Tune_Evaluate"]["const"]["allowed_models"]


class ModelTuner:
    def __init__(self, config_path, allowed_models=None):

        self.factory = ModelFactory(config_path)
        self.config_path = config_path
        self.allowed_models = allowed_models 
        splitter = ScalingWithSplitStrategy()
        self.df = pd.read_csv(preprocessed_data_csv, index_col=[0])
        self.df = self.df.dropna()
        self.df = self.df.drop(columns=['date'])
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = splitter.apply(self.df)
        self.results = []

    def adjusted_r2(self, y_true, y_pred, p):
        try:
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            return adj_r2
        except CustomException as e:
            logger.error(f"Error calculating adjusted R2: {e}")
            return None

    def generate_param_grid(self, param_dict):
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    def evaluate_model(self, model_name, model, X, y):
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
    def run_grid_search(self):
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

            if best_params is None:
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

            # Evaluate final model
            try:
                rmse_train, mae_train, r2_train, adj_r2_train = self.evaluate_model(model_name, final_model, self.X_train, self.y_train)
                rmse_val, mae_val, r2_val, adj_r2_val = self.evaluate_model(model_name, final_model, self.X_val, self.y_val)
                rmse_test, mae_test, r2_test, adj_r2_test = self.evaluate_model(model_name, final_model, self.X_test, self.y_test)
                self.results.append({
                    "model": model_name,
                    "final_model": final_model,
                    "best_params": best_params,
                    "scores": {
                        "train": {"rmse": rmse_train, "mae": mae_train, "r2": r2_train, "adjusted_r2": adj_r2_train},
                        "val": {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "adjusted_r2": adj_r2_val},
                        "test": {"rmse": rmse_test, "mae": mae_test, "r2": r2_test, "adjusted_r2": adj_r2_test},
                    }
                })
            
            except CustomException as e:
                logger.error(f"Error during Evaluation for {model_name}: {e}")

    @track_performance
    def save_results(self, output_path):
        try:
            sorted_results = sorted(self.results, key=lambda x: x['scores']['test']['adjusted_r2'], reverse=True)
            cleaned_results = {}

            for item in sorted_results:
                model_class_name = item["final_model"].__class__.__name__
                params = item["best_params"]

                def clean_scores(scores_dict):
                    return {
                        split: {metric: round(float(value), 4) for metric, value in scores.items()}
                        for split, scores in scores_dict.items()
                    }

                cleaned_results[model_class_name] = {
                    "model": model_class_name,
                    "params": params,
                    "scores": clean_scores(item["scores"])
                }

            # Log the best model and its params
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
    try:

        logger.info("Starting Model Selection and Tuning...")
        tuner = ModelTuner(config_path=regressors_yaml, allowed_models=allowed_models)
        tuner.run_grid_search()
        tuner.save_results(tuned_model_yaml)

        logger.info("Selected model tuning completed successfully.")

    except CustomException as e:
        logger.error(f"Unexpected error in main(): {e}")


if __name__ == "__main__":
    try:

        logger.info("Starting Model Selection and Tuning...")
        execute_model_tune_evaluate()
        logger.info("Selected model tuning completed successfully.")

    except CustomException as e:
        logger.error(f"Unexpected error in main(): {e}")
        
