import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from abc import ABC, abstractmethod
from typing import List, Optional
from Common_Utils import setup_logger, track_performance, CustomException
# Setup logger
logger = setup_logger(filename="logs")

# Step 1: Abstract Base Class
class ScalerStrategy(ABC):
    @abstractmethod
    def get_scaler(self):
        pass

# Step 2: Concrete Implementations
class StandardScalerStrategy(ScalerStrategy):
    def get_scaler(self):
        return StandardScaler()

class RobustScalerStrategy(ScalerStrategy):
    def get_scaler(self):
        return RobustScaler()

# Step 3: Scaler Factory
class ScalerFactory:
    @staticmethod
    def get_scaler(strategy: str) -> ScalerStrategy:
        strategy = strategy.lower()
        if strategy == "standard":
            return StandardScalerStrategy().get_scaler()
        elif strategy == "robust":
            return RobustScalerStrategy().get_scaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")

class ScalingWithSplitStrategy():
    def __init__(self, method='standard', target_column: Optional[str] = 'target',
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.scaler = ScalerFactory.get_scaler(method)
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        #df = df.dropna(subset=[self.target_column]).reset_index(drop=True)
        total_len = len(df)
        train_end = int(total_len * self.train_ratio)
        val_end = train_end + int(total_len * self.val_ratio)

        df['split'] = 'test'
        df.loc[:train_end-1, 'split'] = 'train'
        df.loc[train_end:val_end-1, 'split'] = 'val'
        return df

    @track_performance
    def apply(self,df: pd.DataFrame):
        try:
            df = self.split_data(df)

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if self.target_column:
                numeric_cols = [col for col in numeric_cols if col not in [self.target_column]]

            train_df = df[df['split'] == 'train']
            val_df = df[df['split'] == 'val']
            test_df = df[df['split'] == 'test']

            self.scaler.fit(train_df[numeric_cols])

            df.loc[train_df.index, numeric_cols] = self.scaler.transform(train_df[numeric_cols])
            df.loc[val_df.index, numeric_cols] = self.scaler.transform(val_df[numeric_cols])
            df.loc[test_df.index, numeric_cols] = self.scaler.transform(test_df[numeric_cols])

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
            raise
