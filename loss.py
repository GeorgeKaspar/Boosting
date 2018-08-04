import numpy as np


class BernoulliLoss:
    def __init__(self, nclasses):
        assert nclasses == 2

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = 1
            sample_sum = y.shape[0]
        else:
            sample_sum = np.sum(sample_weight)

        return np.sum(sample_weight * np.log(1 + np.exp(-2.0 * y * y_pred))) / sample_sum

    def negative_gradient(self, y, y_pred, **kwargs):
        e = np.exp(-2.0 * y * y_pred)
        return 2.0 * y * e / (1.0 + e)


class CARTBernoulliLoss(BernoulliLoss):
    def __init__(self, nclasses):
        super(CARTBernoulliLoss, self).__init__(nclasses)

    def update_terminal_regions(self, tree, X, y, y_pred, g, mask, lr=1.0):

        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~mask] = -1

        for leaf in tree._tree.leaves():
            self._update_terminal_region(tree, masked_terminal_regions, leaf, X, y, y_pred, g, lr)

        y_pred += tree._tree._leaf_values.take(terminal_regions, axis=0)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, y_pred, g, lr):

        terminal_region = np.where(terminal_regions == leaf)[0]
        if not terminal_region.size:
            return

        g = g.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        y_pred = y_pred.take(terminal_region, axis=0)

        e = np.exp(-2.0 * y * y_pred)
        g_2 = 4 * y * y * e / ((1 + e)**2)

        tree._tree._leaf_values[leaf] = lr * np.sum(g) / np.sum(g_2)


class SklearnBernoulliLoss(BernoulliLoss):
    def __init__(self, nclasses):
        super(CARTBernoulliLoss, self).__init__(nclasses)

    def update_terminal_regions(self, tree, X, y, y_pred, g, mask, lr=1.0):

        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~mask] = -1

        from sklearn.tree._tree import TREE_LEAF
        for leaf in np.where(tree.tree_.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions, leaf, X, y, y_pred, g, lr)

        y_pred += tree.tree_.value[:, 0, 0].take(terminal_regions, axis=0)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, y_pred, g, lr):

        terminal_region = np.where(terminal_regions == leaf)[0]
        if not terminal_region.size:
            return

        g = g.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        y_pred = y_pred.take(terminal_region, axis=0)

        e = np.exp(-2.0 * y * y_pred)
        g_2 = 4 * y * y * e / ((1 + e)**2)

        tree.tree_.value[leaf] = lr * np.sum(g) / np.sum(g_2)


class ObliviousBernoulliLoss(BernoulliLoss):
    def __init__(self, nclasses):
        super(ObliviousBernoulliLoss, self).__init__(nclasses)

    def update_terminal_regions(self, tree, X, y, y_pred, g, mask, lr=1.0):

        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~mask] = -1

        for leaf in range(tree._tree.num_of_leaves()):
            self._update_terminal_region(tree, masked_terminal_regions, leaf, X, y, y_pred, g, lr)

        y_pred += tree._tree.values.take(terminal_regions, axis=0)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, y_pred, g, lr):

        terminal_region = np.where(terminal_regions == leaf)[0]
        if not terminal_region.size:
            return

        g = g.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        y_pred = y_pred.take(terminal_region, axis=0)

        e = np.exp(-2.0 * y * y_pred)
        g_2 = 4 * y * y * e / ((1 + e)**2)

        tree._tree.values[leaf] = lr * np.sum(g) / np.sum(g_2)
