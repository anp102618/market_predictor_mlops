import os 
import pandas as pd 
import numpy as np
from Common_Utils import setup_logger, track_performance, CustomException, execute_files_backup
from src.Data_Ingestion import execute_data_ingestion
from src.Data_Validation import execute_data_validation
from src.Data_Preprocessing import execute_data_preprocessing
from src.Model_Tune_Evaluate import execute_model_tune_evaluate
from src.Experiment_Tracking_Prediction import execute_mlflow_steps
from Common_Utils import run_dvc_command


logger = setup_logger(filename="logs")

@track_performance
def execute_pipeline():
    try:
        logger.info(" Starting End to End ML pipeline execution ...")
        
        execute_files_backup()
        execute_data_ingestion()
        execute_data_validation()
        execute_data_preprocessing()
        execute_model_tune_evaluate()
        
        # After your pipeline
        run_dvc_command("dvc add Data/processed_data/final_data.csv")
        run_dvc_command("dvc add Tuned_Model/model.joblib")
        run_dvc_command("dvc add Tuned_Model/mlflow_details.yaml")
        run_dvc_command("dvc add Tuned_Model/time_series_predictions.yaml")

        run_dvc_command("git add .")
        run_dvc_command("git commit -m 'Auto: Tracked artifacts with DVC'")
        run_dvc_command("dvc push")  # Will push to local remote ~/dvc_storage

        logger.info("End to End ML pipeline execution completed successfully...")

    except CustomException as e:
        logger.error(f"Unexpected error in End to End ML pipeline execution : {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting End to End ML pipeline execution...")

        execute_pipeline()

        logger.info(" End to End ML pipeline execution completed successfully...")
        
    
    except CustomException as e:
            logger.error(f"Unexpected error in ML pipeline execution : {e}")
    


        




