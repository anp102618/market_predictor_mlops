import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "__init__.py",
    ".github/workflows/.gitkeep",
    ".github/workflows/ci-cd.yaml",
    ".gitignore",
    f"Common_Utils/__init__.py",

    f"Data/__init__.py",
    f"Data/data.db",
    f"Data/raw_data/__init__.py",
    f"Data/new_data/__init__.py",
    f"Data/processed_data/__init__.py",
    f"Data/previous_data/__init__.py",
    f"Data/ref_data/__init__.py",

    f"DataBase/__init__.py",
    f"DataBase/db_handler.py",

    f"Tuned_Model/__init__.py",
    
    f"Config_Yaml/__init__.py",
    f"Config_Yaml/model_path.yaml",
    f"Config_Yaml/model_config.yaml",
    f"Config_Yaml/regressors.yaml",

    f"Model_Utils/__init__.py",
    f"Model_Utils/feature_nan_imputation.py",
    f"Model_Utils/feature_outlier_handling.py",
    f"Model_Utils/regressor_models.py",
    f"Model_Utils/feature_scaling.py",
    f"Model_Utils/feature_selection_extraction.py",
    f"Model_Utils/finbert_implementation.py",
    f"Model_Utils/feature_splitting_scaling.py",
    f"Model_Utils/time_series_models.py",

    f"src/__init__.py",
    f"src/Data_Ingestion.py",
    f"src/Data_Validation.py",
    f"src/Data_Preprocessing.py",
    f"src/Model_tune_evaluate.py",
    f"src/Experiment_Tracking_Prediction.py",

    f"Tuned_Model/__init__.py",

    f"Test_Script/__init__.py",
    f"Test_Script/test_model_promotion.py",

    f"prometheus/__init__.py",
    f"prometheus/prometheus.yml",
    f"prometheus/prom_grafa.txt",
    
    "app.py",
    "requirements.txt",
    "app_requirements.txt",
    "setup.py",
    "main.py",
    "Dockerfile",
    ".dockerignore",
    
 


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")