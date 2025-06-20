import pandas as pd
import numpy as np
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml
from Model_Utils.feature_nan_imputation import ImputerFactory
from Model_Utils.feature_outlier_handling import OutlierHandler
from DataBase.db_handler import SQLiteEditor

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml")

# ------------------ Config Paths ------------------ #
db_path: Path = Path(config["Data_Preprocessing"]["path"]["database"])
final_data_table: str = config["Data_Preprocessing"]["path"]["final_data_table"]
preprocessed_csv: Path = Path(config["Data_Preprocessing"]["path"]["preprocessed_csv"])

# ------------------ Config Constants ------------------ #
date_column: str = config["Data_Preprocessing"]["const"]["date_column"]
imputation_method: str = config["Data_Preprocessing"]["const"]["imputation_method"]
outlier_method: str = config["Data_Preprocessing"]["const"]["outlier_method"]
iqr_threshold: float = config["Data_Preprocessing"]["const"]["iqr_threshold"]
source_column: str = config["Data_Preprocessing"]["const"]["source_column"]
target_column: str = config["Data_Preprocessing"]["const"]["target_column"]


class PreprocessingStrategy(ABC):
    """Abstract base class for all preprocessing strategies."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RemoveDuplicatesStrategy(PreprocessingStrategy):
    """Removes duplicate rows from the DataFrame."""

    @track_performance
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            logger.info(f"Removed {before - after} duplicate rows.")
            return df
        
        except CustomException as e:
            logger.error(f"Error in RemoveDuplicatesStrategy: {e}")
            


class DateFormatConversionStrategy(PreprocessingStrategy):
    """Converts specified column to datetime format."""

    def __init__(self, date_column: str):
        self.date_column = date_column

    @track_performance
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            if df[self.date_column].isnull().any():
                logger.warning("Some date parsing failed and were converted to NaT.")
            return df
        
        except CustomException as e:
            logger.error(f"Error in DateFormatConversionStrategy: {e}")
            


class ImputeMissingValuesStrategy(PreprocessingStrategy):
    """Imputes missing numeric values using a specified method (mean, median, etc.)."""

    def __init__(self, method: str = 'mean'):
        self.imputer = ImputerFactory.get_imputer(method)

    @track_performance
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            logger.info(f"Applied missing value imputation using method: {self.imputer.__class__.__name__}")
            return df
        except CustomException as e:
            logger.error(f"Error in ImputeMissingValuesStrategy: {e}")
            


class OutlierTransformStrategy(PreprocessingStrategy):
    """Applies outlier transformation based on the given method (e.g., Yeo-Johnson)."""

    def __init__(self, method: str = 'yeo', iqr_threshold: float = 1.5):
        self.method = method
        self.iqr_threshold = iqr_threshold

    @track_performance
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            handler = OutlierHandler(df, iqr_threshold=self.iqr_threshold)
            outlier_cols = handler.get_outlier_columns()
            if not outlier_cols:
                logger.info("No outlier columns detected.")
                return df.copy()
            df_transformed = handler.transform(self.method)
            logger.info(f"Outlier transformation applied using method: {self.method} on columns: {outlier_cols}")
            return df_transformed
        
        except CustomException as e:
            logger.error(f"Error in OutlierTransformStrategy: {e}")
            


class TargetShiftStrategy(PreprocessingStrategy):
    """Adds percentage change features and creates a shifted target column."""

    def __init__(self, source_column: str = "nsei", target_column: str = "target"):
        self.source_column = source_column
        self.target_column = target_column

    @track_performance
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in df.columns:
                if col != self.target_column and pd.api.types.is_numeric_dtype(df[col]):
                    pct_change_col = f"{col}_pct_chg"
                    df[pct_change_col] = df[col].pct_change() * 100
                    logger.info(f"Added percentage change column '{pct_change_col}'")

            df = df.dropna()
            df[self.target_column] = df[self.source_column].shift(-1)
            logger.info(f"Shifted column '{self.source_column}' to create '{self.target_column}'")

            return df
        
        except CustomException as e:
            logger.error(f"Error in TargetShiftStrategy: {e}")
            


class PreprocessingContext:
    """Manages and runs a pipeline of preprocessing strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.steps: List[PreprocessingStrategy] = []

    def add_step(self, step: PreprocessingStrategy):
        self.steps.append(step)

    def run(self) -> pd.DataFrame:
        try:
            for step in self.steps:
                self.df = step.apply(self.df)
            logger.info("Preprocessing pipeline completed successfully.")
            return self.df
        
        except CustomException as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            


@track_performance
def data_preprocessing_steps(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Executes all preprocessing steps on the input DataFrame."""
    try:
        context = PreprocessingContext(df)
        context.add_step(RemoveDuplicatesStrategy())
        context.add_step(DateFormatConversionStrategy(date_column=date_column))
        context.add_step(ImputeMissingValuesStrategy(imputation_method))
        context.add_step(OutlierTransformStrategy(outlier_method, iqr_threshold=iqr_threshold))
        context.add_step(TargetShiftStrategy(source_column=source_column, target_column=target_column))

        return context.run()
    
    except CustomException as e:
        logger.error(f"Data Preprocessing process failed: {e}")
        


def execute_data_preprocessing():
    """Main driver function to execute the preprocessing pipeline from database to CSV."""
    try:
        logger.info("Starting Data Preprocessing pipeline")
        db = SQLiteEditor(db_path=db_path)
        df = db.fetch_df(query=f"SELECT * FROM {final_data_table}")
        intermediate_df = data_preprocessing_steps(df)
        intermediate_df.to_csv(preprocessed_csv, index=False)
        logger.info("Data Preprocessing pipeline successfully executed")
    
    except CustomException as e:
        logger.error(f"Data Preprocessing process failed: {e}")
        


if __name__ == "__main__":
    try:
        logger.info("Starting Data Preprocessing pipeline")
        execute_data_preprocessing()
        logger.info("Data Preprocessing pipeline successfully executed")
    
    except CustomException as e:
        logger.error(f"Data Preprocessing process failed: {e}")
        
