from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import argparse
import numpy as np
import blending
import boosting
import matplotlib.pyplot as plt
from loss import BernoulliLoss


def report(y_true, y_pred, name):
    print(name)
    print(classification_report(y_true, np.array(y_pred > 0.5, dtype=np.float32)))
    print('roc_auc: ', roc_auc_score(y_true, y_pred))
    print('log loss: ', log_loss(y_true, y_pred))
    print('***')


def main(pathToTrain, pathToTest):
    dataTrain = np.genfromtxt(pathToTrain, delimiter=' ')
    dataTest = np.genfromtxt(pathToTest, delimiter=' ')

    X_train = dataTrain[:, 1:]
    y_train = dataTrain[:, 0]
    X_test = dataTest[:, 1:]
    y_test = dataTest[:, 0]

    print("shapes: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    '''
    clf = blending.Blending(
        boosting.GradientBoosting(T=100, use_growing_depth=True, subsample=0.9, lr=0.1, max_depth=4),
        [
            MLPClassifier(hidden_layer_sizes=(2,), max_iter=100),
            LogisticRegression(),
        ]
    )
    sklearn_clf = blending.Blending(
        LogisticRegression(),
        [
            GradientBoostingClassifier(criterion='mse', n_estimators=100, presort=True, subsample=0.9, learning_rate=0.1, max_depth=4),
            MLPClassifier(hidden_layer_sizes=(2,), max_iter=400)
        ]
    )
    '''
    sklearn_clf = GradientBoostingClassifier(
        criterion='mse',
        n_estimators=200,
        presort=True,
        subsample=0.9,
        max_depth=4,
        learning_rate=0.1
    )
    clf = boosting.GradientBoosting(
        T=200,
        use_growing_depth=False,
        # use_growing_depth=True,
        subsample=0.9,
        max_depth=4,
        lr=0.1
    )
    '''
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_0 = clf.models[0].predict(X_test)
    y_pred_1 = clf.models[1].predict(X_test)
    '''
    y_gen = clf.staged_predict(X_test)
    sklearn_y_gen = sklearn_clf.staged_predict(X_test)

    self_loss = []
    sklearn_loss = []
    for y_pred, sklearn_y_pred in zip(y_gen, sklearn_y_gen):
        self_loss.append(log_loss(y_test, y_pred))
        sklearn_loss.append(log_loss(y_test, sklearn_y_pred))
    self_plot, = plt.plot(self_loss, label='self')
    sklearn_plot, = plt.plot(sklearn_loss, label='sklearn')
    plt.legend(handles=[self_plot, sklearn_plot])
    plt.show()
    # '''
    report(y_test, y_pred, "blended")
    report(y_test, y_pred_0, "gradient boosting")
    report(y_test, y_pred_1, "neural net")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pathToTrain')
    parser.add_argument('pathToTest')
    args = parser.parse_args()
    main(args.pathToTrain, args.pathToTest)
