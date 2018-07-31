import numpy as np
import estimators
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, log_loss, classification_report, mean_squared_error
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

def main():
    X = np.load('X.npy')
    y = np.load('g.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(X_train.shape, y_train.shape)

    for depth in range(1,16):
        clf = estimators.DecisionTreeRegressor(max_depth=depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        sklearn_clf = DecisionTreeRegressor(max_depth=depth) 
        sklearn_clf.fit(X_train, y_train)
        sklearn_y_pred = sklearn_clf.predict(X_test)

        print("self", mean_squared_error(y_test, y_pred))
        print("sklearn", mean_squared_error(y_test, sklearn_y_pred))

    return 0

if __name__ == '__main__':
    main()
