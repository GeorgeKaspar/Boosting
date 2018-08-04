import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from loss import BernoulliLoss
from estimators import ObliviousDecisionTreeRegressor, DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
from numba import jit
from functools import reduce


class GradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, T=10, lr=1e-1, max_depth=6, use_growing_depth=False, subsample=1.0):
        self.lr = lr
        self.T = T
        self.max_depth = max_depth
        self.use_growing_depth = use_growing_depth
        self.subsample = subsample
        self.loss = BernoulliLoss(2)

    def fit(self, X, y):
        self.trees = []

        X_idx_sorted = np.array(np.argsort(X, axis=0), dtype=np.int32)

        tree_0 = DecisionTreeClassifier(max_depth=self.max_depth)
        tree_0.fit(X, y.astype('int64'), X_idx_sorted=X_idx_sorted)
        self.trees.append(tree_0)
        y_pred = 2 * tree_0.predict_proba(X)[:, 1].astype('float64') - 1
        y = 2 * y - 1

        for i in range(self.T):
            print("building tree %d" % (i + 1))
            mask = (np.random.random(size=X.shape[0]) < self.subsample) if self.subsample < 1.0 else np.ones(X.shape[0], dtype=np.bool)
            g = self.loss.negative_gradient(y, y_pred)
            _tree = ObliviousDecisionTreeRegressor(
                max_depth=1 + int(i * self.max_depth / self.T) if self.use_growing_depth else self.max_depth
            )
            _tree.fit(X, g, X_idx_sorted=X_idx_sorted, sample_weight=mask)
            self.loss.update_terminal_regions(_tree, X, y, y_pred, g, mask, self.lr)
            self.trees.append(_tree)

    def predict(self, X):
        return (reduce(lambda value, tree: value + tree.predict(X), self.trees, 0.0) + 1) / 2

    def staged_predict(self, X):
        pred = 0
        for i in range(len(self.trees)):
            pred += self.trees[i].predict(X).astype('float64')
            yield pred
