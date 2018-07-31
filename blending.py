import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

class Blending(BaseEstimator):
    def __init__(self, metamodel=None, models=None, test_size=0.5):
        super(Blending, self).__init__()
        self.models = models
        self.metamodel = metamodel
        self.test_size = test_size

    def fit(self, X, y):
        X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=self.test_size)
        for i in range(len(self.models)):
            self.models[i].fit(X_train, y_train)
        
        print(X_meta.shape, y_meta.shape)
        y_meta_pred = np.array(list(map(lambda x : x.predict(X_meta), self.models))).T
        self.metamodel.fit(y_meta_pred, y_meta)

    def predict(self, X):
        y_meta = np.array(list(map(lambda x : x.predict(X), self.models))).T
        return self.metamodel.predict(y_meta)
    
    def __repr__(self):
        return "Blending(metamodel=" + self.metamodel.__repr__() + '), models = [' + ', '.join(list(map(lambda x : x.__repr__(), self.models))) + '])'

    def __str__(self):
        return  self.__repr__()

