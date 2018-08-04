import numpy as np
from collections import namedtuple

split_criterion = namedtuple('split_criterion', ['feature', 'value', 'gain'])


class BinaryDecisionTree(object):
    def __init__(self, n_features):
        self._capacity = 0

        self._n_features = n_features
        self._is_leaf = np.zeros(0, dtype='bool')
        self._is_node = np.zeros(0, dtype='bool')
        self._leaf_values = np.zeros(0)
        self._leaf_n_samples = np.zeros(0)
        self._splits = []

        self._capacity = 0
        self._reallocate_if_needed(required_capacity=1)
        self._init_root()

    def _reallocate_if_needed(self, required_capacity):
        if self._capacity <= required_capacity:
            self._is_leaf.resize(required_capacity)
            self._is_node.resize(required_capacity)
            self._leaf_values.resize(required_capacity)
            self._leaf_n_samples.resize(required_capacity)
            self._splits = self._grow_list(self._splits, required_capacity)
            self._capacity = required_capacity

    def _init_root(self):
        self._is_leaf[0] = True
        self._is_node[0] = True
        self._latest_used_node_id = 0

    def num_of_leaves(self):
        return np.sum(self._is_leaf[:self._latest_used_node_id + 1])

    def num_of_nodes(self):
        return self._latest_used_node_id

    def is_leaf(self, node_id):
        assert node_id >= 0 and node_id <= self._latest_used_node_id
        return self._is_leaf[node_id]

    def leaf_mask(self):
        return self._is_leaf[:self._latest_used_node_id + 1]

    def left_child(self, node_id):
        return (node_id + 1) * 2 - 1

    def right_child(self, node_id):
        return (node_id + 1) * 2

    def leaves(self):
        return np.where(self._is_leaf)[0]

    def split_node(self, node_id, split):
        left_child_id = self.left_child(node_id)
        right_child_id = self.right_child(node_id)

        if right_child_id >= self._capacity:
            self._reallocate_if_needed(2 * self._capacity + 1)

        self._splits[node_id] = deepcopy(split)
        self._is_leaf[node_id] = False
        self._is_node[left_child_id] = True
        self._is_node[right_child_id] = True
        self._is_leaf[left_child_id] = True
        self._is_leaf[right_child_id] = True
        self._latest_used_node_id = max(self._latest_used_node_id, right_child_id)

    def predict(self, X):
        def predict_one(x):
            current_node = self.root()
            while not self.is_leaf(current_node):
                current_split = self._splits[current_node]
                if x[current_split.feature_id] < current_split.value:
                    current_node = self.left_child(current_node)
                else:
                    current_node = self.right_child(current_node)
            return self._leaf_values[current_node]

        return np.apply_along_axis(predict_one, 1, X)

    def apply(self, X):
        def apply_one(x):
            current_node = self.root()
            while not self.is_leaf(current_node):
                current_split = self._splits[current_node]
                if x[current_split.feature_id] < current_split.value:
                    current_node = self.left_child(current_node)
                else:
                    current_node = self.right_child(current_node)
            return current_node

        sample_size, features_count = X.shape
        result = np.zeros(sample_size, dtype=np.int64)
        for i in range(sample_size):
            x = X[i]
            result[i] = apply_one(x)
        return result

    def root(self):
        return 0

    def depth(self, node_id):
        return np.floor(np.log2(node_id + 1)) + 1

    def _grow_list(self, lst, required_capacity, fill_value=None):
        if len(lst) >= required_capacity:
            return lst
        return lst + [fill_value for _ in range(required_capacity - len(lst))]


class TreeSplitCART:
    def __init__(self, feature_id, value, gain):
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


class TreeBuilderCART(object):
    EPS = 0.1

    def __init__(self, max_depth=10, max_features=None, min_samples_per_leaf=2, bins=40, **kwargs):
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_features = max_features
        self.bins = bins
        if max_features == 'sqrt':
            self.get_feature_ids = self.__get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.__get_feature_ids_log2
        elif max_features is None:
            self.get_feature_ids = self.__get_feature_ids_N
        else:
            print('invalid max_features name')
            raise

    def __get_feature_ids_sqrt(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[0:int(np.sqrt(n_feature))]

    def __get_feature_ids_log2(self, n_feature):
        feature_ids = range(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[0:int(np.log2(n_feature))]

    def __get_feature_ids_N(self, n_feature):
        feature_ids = range(n_feature)
        return feature_ids[0:n_feature]

    def build_tree(self, X, y, X_idx_sorted):
        n_samples, n_features = X.shape
        tree = BinaryDecisionTree(n_features=n_features)
        leaf_to_split = tree.root()
        self._build_tree_recursive(tree, leaf_to_split, X, y, X_idx_sorted, np.ones(X.shape[0], dtype=np.bool))
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y, X_idx_sorted, mask):
        n_samples = np.sum(mask)
        if n_samples < 1:
            return

        leaf_reached = False

        if n_samples <= 2 * self.min_samples_per_leaf + 2:
            leaf_reached = True

        depth = tree.depth(cur_node)
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_reached = True

        best_split = None
        if not leaf_reached:
            best_split = self.find_best_split(X, y, X_idx_sorted, mask)
            if best_split is None:
                leaf_reached = True

        tree._leaf_n_samples[cur_node] = np.sum(mask)
        if leaf_reached:
            tree._leaf_values[cur_node] = np.mean(y[mask])
        else:
            tree.split_node(cur_node, best_split)

            left_child = tree.left_child(cur_node)
            right_child = tree.right_child(cur_node)

            mask_left, mask_right = split_dataset(X, y, X_idx_sorted, mask, best_split.feature_id, best_split.value)
            self._build_tree_recursive(tree, left_child, X, y, X_idx_sorted, mask_left)
            self._build_tree_recursive(tree, right_child, X, y, X_idx_sorted, mask_right)
        return self

    def find_best_split(self, X, y, X_idx_sorted, mask):
        n_samples, n_features = X.shape
        best_split = None
        for feature_id in self.get_feature_ids(n_features):
            idx = X_idx_sorted[:, feature_id]
            x_f = X[:, feature_id].take(idx, axis=0)
            y_f = y.take(idx, axis=0)
            mask_f = mask.take(idx, axis=0)

            x_mf = x_f[mask_f]
            y_mf = y_f[mask_f]

            s2 = np.sum(y_mf**2)
            s = np.sum(y_mf)
            n = y_mf.shape[0]

            s2_left = np.cumsum(y_mf**2)[:-1]
            s2_right = s2 - s2_left

            s_left = np.cumsum(y_mf)[:-1]
            s_right = s - s_left

            n_left = np.arange(1, s_left.shape[0] + 1)
            n_right = (n - n_left)

            gain = n_left * n_right * ((s_left / n_left - s_right / n_right)**2) / n

            ind = np.argmax(gain[self.min_samples_per_leaf:-self.min_samples_per_leaf]) + self.min_samples_per_leaf
            ind = self._grouped(ind, x_mf)
            if ind is None:
                continue

            value = (x_mf[ind] + x_mf[ind + 1]) / 2.0

            gain = gain[ind]

            if gain >= 0:
                split = TreeSplitCART(gain=gain, feature_id=feature_id, value=value)
                if best_split is None or (split.gain > best_split.gain):
                    best_split = split
        return best_split

    def _grouped(self, ind, y):
        pad = self.min_samples_per_leaf
        idx = np.where(np.abs(np.diff(y[pad:-pad])) > 1e-3)[0] + pad
        if len(idx) == 0:
            return None
        return idx[np.argmin(np.abs(idx - ind))]


class ObliviousTree(object):
    def __init__(self, n_features):
        super(ObliviousTree, self).__init__()
        self.n_features = n_features
        self.split_criterions = []
        self.values = None

    def num_of_leaves(self):
        return 2 ** len(self.split_criterions)

    def is_fitted(self):
        return self.values is not None

    def predict(self, X):
        '''
        X [n_samples x n_features]
        '''
        ind = self.apply(X)
        return self.values[ind]

    def apply(self, X):
        assert self.is_fitted()
        assert self.n_features == X.shape[1]
        ind = np.zeros(X.shape[0])
        for split in self.split_criterions:
            ind = 2 * ind + (X[:, split.feature] < split.values)
        return ind

    def depth(self, cur_node=None):
        if cur_node is None:
            return len(self.split_criterions)

        return cur_node + 1

    def add_split(self, feature, value):
        self.split_criterions.append(split_criterion(feature=feature, value=value))


def ObliviousTreeBuilder(object):

    EPS = 0.1

    def __init__(self, max_depth=10, max_features=None, min_samples_per_leaf=2, **kwargs):
        super(ObliviousTreeBuilder, self).__init__()
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_features = max_features
        if max_features == 'sqrt':
            self.get_feature_ids = self.__get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.__get_feature_ids_log2
        elif max_features is None:
            self.get_feature_ids = self.__get_feature_ids_N
        else:
            print('invalid max_features name')
            raise

    def __get_feature_ids_sqrt(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[0:int(np.sqrt(n_feature))]

    def __get_feature_ids_log2(self, n_feature):
        feature_ids = range(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[0:int(np.log2(n_feature))]

    def __get_feature_ids_N(self, n_feature):
        feature_ids = range(n_feature)
        return feature_ids[0:n_feature]

    def find_best_split(self, X, y, X_idx_sorted, mask):
        n_samples, n_features = X.shape
        best_split = None
        for feature_id in self.get_feature_ids(n_features):
            idx = X_idx_sorted[:, feature_id]
            x_f = X[:, feature_id].take(idx, axis=0)
            y_f = y.take(idx, axis=0)
            mask_f = mask.take(idx, axis=0)

            for i in range(2 ** cur_depth):
                mask_f_node = (mask_f == i)
                x_mf = x_f[mask_f_node]
                y_mf = y_f[mask_f_node]
                s2 = np.sum(y_mf**2)
                s = np.sum(y_mf)
                n = y_mf.shape[0]
                s2_left = np.cumsum(y_mf**2)[:-1]
                s2_right = s2 - s2_left
                s_left = np.cumsum(y_mf)[:-1]
                s_right = s - s_left
                n_left = np.arange(1, s_left.shape[0] + 1)
                n_right = (n - n_left)
                gain = n_left * n_right * ((s_left / n_left - s_right / n_right)**2) / n
                ind = np.argmax(gain[self.min_samples_per_leaf:-self.min_samples_per_leaf]) + self.min_samples_per_leaf
                ind = self._grouped(ind, x_mf)
                if ind is None:
                    continue

                value = (x_mf[ind] + x_mf[ind + 1]) / 2.0

                gain = gain[ind]

                if gain >= 0:
                    split = split_criterion(gain=gain, feature=feature_id, value=value)
                    if best_split is None or (split.gain > best_split.gain):
                        best_split = split
        return best_split

    def _grouped(self, ind, y):
        pad = self.min_samples_per_leaf
        idx = np.where(np.abs(np.diff(y[pad:-pad])) > 1e-3)[0] + pad
        if len(idx) == 0:
            return None
        return idx[np.argmin(np.abs(idx - ind))]

    def split_dataset(self, X, mask, split, cur_depth):
        mask_cur = (X[:, split.feature] >= split.value)
        mask[mask_cur] += 2 ** cur_depth
        return mask

    def _build_tree_recursive(self, tree, cur_depth, X, y, X_idx_sorted, mask):
        leaf_reached = False

        depth = tree.depth(cur_node)
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_reached = True
        _, node_counts = np.unique(mask)
        if np.any(counts <= 2 * self.min_samples_per_leaf + 2):
            leaf_reached = True
        if leaf_reached:
            tree.values = np.zeros(2 ** cur_depth, dtype=np.float32)
            tree.values[mask] += y
            tree.values /= node_counts
            return tree

        best_split = self.find_best_split(X, y, X_idx_sorted, mask)
        if best_split is None:
            leaf_reached = True

        if leaf_reached:
            tree.values = np.zeros(2 ** cur_depth, dtype=np.float32)
            tree.values[mask] += y
            tree.values /= node_counts
            return tree

        tree.add_split(best_split)
        mask = self.split_dataset(X, mask, best_split, cur_depth)
        tree = self._build_tree_recursive(tree, cur_depth + 1, X, y, X_idx_sorted, mask)

        return tree

    def build_tree(self, X, y, X_idx_sorted):
        n_samples, n_features = X.shape
        tree = ObliviousTree(n_features=n_features)
        cur_depth = 0
        tree = self._build_tree_recursive(tree, cur_depth, X, y, X_idx_sorted, np.zeros(X.shape[0], dtype=np.bool))
        return tree
