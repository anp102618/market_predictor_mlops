from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from abc import ABC, abstractmethod

# Abstract base strategy
class ImputerStrategy(ABC):
    @abstractmethod
    def get_imputer(self):
        pass

# Mean Imputer
class MeanImputerStrategy(ImputerStrategy):
    def get_imputer(self):
        return SimpleImputer(strategy="mean")

# Median Imputer
class MedianImputerStrategy(ImputerStrategy):
    def get_imputer(self):
        return SimpleImputer(strategy="median")

# KNN Imputer
class KNNImputerStrategy(ImputerStrategy):
    def get_imputer(self):
        return KNNImputer(n_neighbors=5)

# Iterative Imputer
class IterativeImputerStrategy(ImputerStrategy):
    def get_imputer(self):
        return enable_iterative_imputer(random_state=42)

# Factory Class
class ImputerFactory:
    @staticmethod
    def get_imputer(strategy: str):
        strategy = strategy.lower()
        if strategy == "mean":
            return MeanImputerStrategy().get_imputer()
        elif strategy == "median":
            return MedianImputerStrategy().get_imputer()
        elif strategy == "knn":
            return KNNImputerStrategy().get_imputer()
        elif strategy == "iterative":
            return IterativeImputerStrategy().get_imputer()
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
