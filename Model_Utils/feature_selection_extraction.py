from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from typing import Optional, Union


class FeatureProcessor(ABC):
    """
    Abstract base class for feature selection or extraction strategies.
    """
    @abstractmethod
    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply the feature processing strategy.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (Optional[pd.Series]): Target variable if required.

        Returns:
            pd.DataFrame: Transformed feature matrix.
        """
        pass


# ========================== Selection Strategies ===========================

class SelectKBestStrategy(FeatureProcessor):
    """
    Selects top-k features based on univariate regression tests.
    """
    def __init__(self, k: int = 10):
        self.k = k
        self.selector = SelectKBest(score_func=f_regression, k=self.k)

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if y is None:
            raise ValueError("Target variable 'y' is required for SelectKBest.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.selector.fit_transform(X, y)
        selected_features = X.columns[self.selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features, index=X.index)


class VarianceThresholdStrategy(FeatureProcessor):
    """
    Removes features with variance below a certain threshold.
    """
    def __init__(self, threshold: float = 0.0):
        self.selector = VarianceThreshold(threshold=threshold)

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.selector.fit_transform(X)
        selected_features = X.columns[self.selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features, index=X.index)


# ========================== Extraction Strategies ===========================

class LDAStrategy(FeatureProcessor):
    """
    Applies Linear Discriminant Analysis for dimensionality reduction.
    """
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.lda = LDA(n_components=self.n_components)

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if y is None:
            raise ValueError("Target variable 'y' is required for LDA.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.lda.fit_transform(X, y)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        n_comps = X_new.shape[1]
        columns = [f'LDA{i+1}' for i in range(n_comps)]
        return pd.DataFrame(X_new, columns=columns, index=X.index)


class KernelPCAStrategy(FeatureProcessor):
    """
    Applies Kernel PCA transformation for nonlinear feature extraction.
    """
    def __init__(self, n_components: int = 5, kernel: str = 'rbf'):
        self.n_components = n_components
        self.kernel = kernel
        self.kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel)

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.kpca.fit_transform(X)
        columns = [f'KPC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_new, columns=columns, index=X.index)


# ============================ Strategy Factory ============================

class FeatureFactory:
    """
    Factory class for creating feature processors for selection or extraction.
    """
    @staticmethod
    def get_processor(kind: str, method: str, **kwargs) -> FeatureProcessor:
        """
        Instantiates the requested feature processor.

        Args:
            kind (str): Either 'selection' or 'extraction'.
            method (str): Name of the method to apply.
            kwargs: Additional parameters for the processor.

        Returns:
            FeatureProcessor: An instance of a feature processor.

        Raises:
            ValueError: If an unsupported kind or method is specified.
        """
        kind = kind.lower()
        method = method.lower()

        if kind == 'selection':
            if method == 'selectkbest':
                return SelectKBestStrategy(**kwargs)
            elif method == 'variancethreshold':
                return VarianceThresholdStrategy(**kwargs)
            else:
                raise ValueError(f"Unknown selection method '{method}'")
        
        elif kind == 'extraction':
            if method == 'lda':
                return LDAStrategy(**kwargs)
            elif method == 'kernelpca':
                return KernelPCAStrategy(**kwargs)
            else:
                raise ValueError(f"Unknown extraction method '{method}'")
        
        else:
            raise ValueError(f"Unknown feature processor kind '{kind}'")
