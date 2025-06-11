import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer
from typing import List

# Utility function to find outlier columns by IQR
def find_outlier_columns(df: pd.DataFrame, threshold: float = 1.5) -> List[str]:
    outlier_cols = []
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        if df[(df[col] < lower) | (df[col] > upper)].any().any():
            outlier_cols.append(col)
    return outlier_cols

# Abstract base class for strategies
class OutlierHandlerStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

# Log Transform strategy (only positive values)
class LogTransformStrategy(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        applicable_cols = []
        for col in columns:
            if (df_copy[col] <= 0).any():
                continue  # skip non-positive columns for log transform
            applicable_cols.append(col)
            df_copy[col] = np.log(df_copy[col])
        if not applicable_cols:
            print("No columns suitable for log transform (contain non-positive values).")
        return df_copy

# Yeo-Johnson Transform strategy (works with negatives and zero)
class YeoJohnsonTransformStrategy(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        if not columns:
            print("No columns detected with outliers for Yeo-Johnson transform.")
            return df_copy
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df_copy[columns] = pt.fit_transform(df_copy[columns])
        return df_copy

# Factory to get appropriate handler
class OutlierHandlerFactory:
    @staticmethod
    def get_handler(strategy: str):
        strat = strategy.lower()
        if strat == 'log':
            return LogTransformStrategy()
        elif strat == 'yeo':
            return YeoJohnsonTransformStrategy()
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'log' or 'yeo'.")

# Combined interface class that uses the factory
class OutlierHandler:
    def __init__(self, df: pd.DataFrame, iqr_threshold: float = 1.5):
        self.df = df.copy()
        self.iqr_threshold = iqr_threshold
        self.outlier_columns = find_outlier_columns(self.df, self.iqr_threshold)

    def transform(self, strategy: str) -> pd.DataFrame:
        handler = OutlierHandlerFactory.get_handler(strategy)
        return handler.handle(self.df, self.outlier_columns)

    def get_outlier_columns(self) -> List[str]:
        return self.outlier_columns
