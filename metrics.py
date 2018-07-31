import numpy as np

def compute_split_info(args):
    split_criterion, X, y, feature_id, split_value = args
    _, _, y_left, y_right = split_dataset(X, y, feature_id, split_value)
    n_left, n_right = len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return None, n_left, n_right
    gain = compute_split_gain(split_criterion, y, y_left, y_right)
    return gain, n_left, n_right

def masked_sorted(y, mask, idx):
    y_sorted = y.take(idx, axis=0)
    mask_sorted = mask.take(idx, axis=0)
    masked_idx = idx[mask_sorted]
    masked_y_sorted = y_sorted[mask_sorted]
    return masked_y_sorted, mask_sorted, masked_idx

def split_dataset(X, y, X_idx_sorted, mask, feature_id, value):
    mask_left = (X[:, feature_id] <= value)
    return np.logical_and(mask, mask_left), np.logical_and(mask, np.logical_not(mask_left))


