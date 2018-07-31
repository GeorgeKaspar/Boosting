import numpy as np
from collections import namedtuple

split_criterion = namedtuple('split_criterion', ['feature', 'value', 'gain'])

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
        elif max_features == None:
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


