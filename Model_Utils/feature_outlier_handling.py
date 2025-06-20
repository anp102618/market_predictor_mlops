import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer
from typing import List, Optional


def find_outlier_columns(df: pd.DataFrame, threshold: float = 1.5) -> List[str]:
    """
    Identify numeric columns in the DataFrame that contain outliers
    based on the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The input dataframe.
        threshold (float): IQR threshold to detect outliers.

    Returns:
        List[str]: List of column names with outliers.
    """
    if df.empty:
        return []

    outlier_cols = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR

        if ((df[col] < lower) | (df[col] > upper)).any():
            outlier_cols.append(col)

    return outlier_cols


class OutlierHandlerStrategy(ABC):
    """
    Abstract base class for outlier handling strategies.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies transformation to handle outliers on selected columns.

        Args:
            df (pd.DataFrame): The input dataframe.
            columns (List[str]): List of columns to transform.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        pass


class LogTransformStrategy(OutlierHandlerStrategy):
    """
    Applies logarithmic transformation to positive-valued columns.
    """
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        applicable_cols = []

        for col in columns:
            if (df_copy[col] <= 0).any():
                continue  # log undefined for non-positive values
            applicable_cols.append(col)
            df_copy[col] = np.log(df_copy[col])

        if not applicable_cols:
            print("No columns suitable for log transform (contain non-positive values).")

        return df_copy


class YeoJohnsonTransformStrategy(OutlierHandlerStrategy):
    """
    Applies Yeo-Johnson transformation (handles zero and negative values).
    """
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()

        if not columns:
            print("No columns detected with outliers for Yeo-Johnson transform.")
            return df_copy

        try:
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            df_copy[columns] = transformer.fit_transform(df_copy[columns])
        except Exception as e:
            print(f"Yeo-Johnson transformation failed: {e}")

        return df_copy


class OutlierHandlerFactory:
    """
    Factory for selecting the appropriate outlier handling strategy.
    """
    @staticmethod
    def get_handler(strategy: str) -> OutlierHandlerStrategy:
        strategy = strategy.lower().strip()
        if strategy == 'log':
            return LogTransformStrategy()
        elif strategy == 'yeo':
            return YeoJohnsonTransformStrategy()
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: ['log', 'yeo'].")


class OutlierHandler:
    """
    Interface class that integrates outlier detection and transformation.

    Attributes:
        df (pd.DataFrame): Input data.
        iqr_threshold (float): Threshold for IQR-based outlier detection.
        outlier_columns (List[str]): Columns containing outliers.
    """
    def __init__(self, df: pd.DataFrame, iqr_threshold: float = 1.5):
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input to be a pandas DataFrame.")

        self.df: pd.DataFrame = df.copy()
        self.iqr_threshold: float = iqr_threshold
        self.outlier_columns: List[str] = find_outlier_columns(self.df, self.iqr_threshold)

    def transform(self, strategy: str) -> pd.DataFrame:
        """
        Applies the chosen transformation strategy to handle outliers.

        Args:
            strategy (str): Strategy to use ('log' or 'yeo').

        Returns:
            pd.DataFrame: DataFrame with transformed columns.
        """
        handler = OutlierHandlerFactory.get_handler(strategy)
        return handler.handle(self.df, self.outlier_columns)

    def get_outlier_columns(self) -> List[str]:
        """
        Returns the list of columns identified as containing outliers.

        Returns:
            List[str]: List of outlier column names.
        """
        return self.outlier_columns
