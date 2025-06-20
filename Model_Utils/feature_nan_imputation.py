from abc import ABC, abstractmethod
from typing import Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import TransformerMixin


class ImputerStrategy(ABC):
    """
    Abstract base class for imputation strategies.
    All strategies must implement the `get_imputer` method.
    """
    @abstractmethod
    def get_imputer(self) -> TransformerMixin:
        pass


class MeanImputerStrategy(ImputerStrategy):
    """
    Strategy for imputing missing values using the mean.
    """
    def get_imputer(self) -> SimpleImputer:
        return SimpleImputer(strategy="mean")


class MedianImputerStrategy(ImputerStrategy):
    """
    Strategy for imputing missing values using the median.
    """
    def get_imputer(self) -> SimpleImputer:
        return SimpleImputer(strategy="median")


class KNNImputerStrategy(ImputerStrategy):
    """
    Strategy for imputing missing values using k-Nearest Neighbors.
    """
    def get_imputer(self) -> KNNImputer:
        return KNNImputer(n_neighbors=5)


class IterativeImputerStrategy(ImputerStrategy):
    """
    Strategy for imputing missing values using Iterative Imputation.
    """
    def get_imputer(self) -> enable_iterative_imputer:
        return enable_iterative_imputer(random_state=42)


class ImputerFactory:
    """
    Factory class for creating imputer instances based on strategy name.

    Supported strategies:
        - "mean"
        - "median"
        - "knn"
        - "iterative"
    """
    @staticmethod
    def get_imputer(strategy: str) -> Union[SimpleImputer, KNNImputer]:
        """
        Returns an imputer object based on the selected strategy.

        Args:
            strategy (str): The name of the imputation strategy.

        Returns:
            TransformerMixin: An imputer instance corresponding to the strategy.

        Raises:
            ValueError: If an unsupported strategy is provided.
        """
        strategy = strategy.strip().lower()

        if strategy == "mean":
            return MeanImputerStrategy().get_imputer()
        elif strategy == "median":
            return MedianImputerStrategy().get_imputer()
        elif strategy == "knn":
            return KNNImputerStrategy().get_imputer()
        elif strategy == "iterative":
            return IterativeImputerStrategy().get_imputer()
        else:
            raise ValueError(f"Unknown imputation strategy: '{strategy}'. "
                             f"Supported: ['mean', 'median', 'knn', 'iterative']")
