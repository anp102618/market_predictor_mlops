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
    


        




