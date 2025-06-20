import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from Common_Utils import setup_logger, track_performance, CustomException

# Setup logger
logger = setup_logger(filename="logs")


# ============================ Scaler Strategy =============================

class ScalerStrategy(ABC):
    """
    Abstract base class for scaling strategies.
    """
    @abstractmethod
    def get_scaler(self):
        """
        Return a scikit-learn scaler object.
        """
        pass


class StandardScalerStrategy(ScalerStrategy):
    """
    Standard scaling (mean=0, std=1).
    """
    def get_scaler(self) -> StandardScaler:
        return StandardScaler()


class RobustScalerStrategy(ScalerStrategy):
    """
    Robust scaling (using median and IQR).
    """
    def get_scaler(self) -> RobustScaler:
        return RobustScaler()


class ScalerFactory:
    """
    Factory class to return the appropriate scaler based on strategy name.
    """
    @staticmethod
    def get_scaler(strategy: str):
        strategy = strategy.lower()
        if strategy == "standard":
            return StandardScalerStrategy().get_scaler()
        elif strategy == "robust":
            return RobustScalerStrategy().get_scaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}. Use 'standard' or 'robust'.")


# ============================ Scaling & Splitting =============================

class ScalingWithSplitStrategy:
    """
    Handles train/val/test split and scaling of numeric features using a given strategy.
    """

    def __init__(
        self,
        method: str = 'standard',
        target_column: Optional[str] = 'target',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        self.scaler = ScalerFactory.get_scaler(method)
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a split column to indicate train/val/test partitions.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

        df = df.copy()
        total_len = len(df)
        train_end = int(total_len * self.train_ratio)
        val_end = train_end + int(total_len * self.val_ratio)

        df['split'] = 'test'
        df.loc[df.index[:train_end], 'split'] = 'train'
        df.loc[df.index[train_end:val_end], 'split'] = 'val'

        return df

    @track_performance
    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Applies scaling and returns train/val/test splits.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            df = self.split_data(df)

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)

            train_df = df[df['split'] == 'train']
            val_df = df[df['split'] == 'val']
            test_df = df[df['split'] == 'test']

            if train_df.empty:
                raise ValueError("Training data is empty after split.")

            self.scaler.fit(train_df[numeric_cols])

            df.loc[train_df.index, numeric_cols] = self.scaler.transform(train_df[numeric_cols])
            df.loc[val_df.index, numeric_cols] = self.scaler.transform(val_df[numeric_cols])
            df.loc[test_df.index, numeric_cols] = self.scaler.transform(test_df[numeric_cols])

            # Extract splits
            X_train = df[df['split'] == 'train'].drop(columns=[self.target_column, 'split'])
            y_train = df[df['split'] == 'train'][self.target_column]

            X_val = df[df['split'] == 'val'].drop(columns=[self.target_column, 'split'])
            y_val = df[df['split'] == 'val'][self.target_column]

            X_test = df[df['split'] == 'test'].drop(columns=[self.target_column, 'split'])
            y_test = df[df['split'] == 'test'][self.target_column]

            logger.info("Successfully applied scaling and data splitting.")
            return X_train, X_val, X_test, y_train, y_val, y_test

        except CustomException as e:
            logger.error(f"Error in ScalingWithSplitStrategy: {e}")
            
