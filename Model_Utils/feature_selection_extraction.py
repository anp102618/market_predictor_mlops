from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

class FeatureProcessor(ABC):
    @abstractmethod
    def process(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pass

# Selection strategies
class SelectKBestStrategy(FeatureProcessor):
    def __init__(self, k=10):
        self.k = k
        self.selector = SelectKBest(score_func=f_regression, k=self.k)

    def process(self, X, y):
        if y is None:
            raise ValueError("Target variable y is required for SelectKBest")
        X_new = self.selector.fit_transform(X, y)
        selected_features = X.columns[self.selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

class VarianceThresholdStrategy(FeatureProcessor):
    def __init__(self, threshold=0.0):
        self.selector = VarianceThreshold(threshold=threshold)

    def process(self, X, y=None):
        X_new = self.selector.fit_transform(X)
        selected_features = X.columns[self.selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

# Extraction strategies
class LDAStrategy(FeatureProcessor):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.lda = LDA(n_components=self.n_components)

    def process(self, X, y):
        if y is None:
            raise ValueError("Target variable y is required for LDA")
        X_new = self.lda.fit_transform(X, y)
        n_comps = X_new.shape[1] if len(X_new.shape) > 1 else 1
        columns = [f'LDA{i+1}' for i in range(n_comps)]
        if n_comps == 1:
            X_new = X_new.reshape(-1, 1)
        return pd.DataFrame(X_new, columns=columns)

class KernelPCAStrategy(FeatureProcessor):
    def __init__(self, n_components=5, kernel='rbf'):
        self.n_components = n_components
        self.kernel = kernel
        self.kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel)

    def process(self, X, y=None):
        X_new = self.kpca.fit_transform(X)
        columns = [f'KPC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_new, columns=columns)

# Factory
class FeatureFactory:
    @staticmethod
    def get_processor(kind: str, method: str, **kwargs) -> FeatureProcessor:
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
