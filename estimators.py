import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from tree import ObliviousTreeBuilder, ObliviousTree


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super(DecisionTreeRegressor, self).__init__()
        self._tree = None
        self._tree_builder = ObliviousTreeBuilder(**kwargs)

    def fit(self, X, y, X_idx_sorted=None, **kwargs):
        data_size, n_features = X.shape
        self._n_features = n_features
        if X_idx_sorted is None:
            X_idx_sorted = np.argsort(X, axis=0)
        self._tree = self._tree_builder.build_tree(X, y, X_idx_sorted)
        return self

    def predict(self, X):
        return self._tree.predict(X)
    
    def apply(self, X):
        return self._tree.apply(X)
