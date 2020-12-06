from base_classifier import BaseClassifier
import numpy as np
from sklearn.svm import SVC


class SVMClassifier(BaseClassifier):

    def __init__(self):
        super().__init__('SVM')
        self.hyper_parameters = {}  # dictionary of the chosen hyper-parameters
        self.model = None

    def get_hyper_parameters_grid(self):
        grid = {'C': [0.01, 0.1, 1, 10, 50], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
        return grid

    def set_hyper_parameters(self, hyper_parameters_dict):
        self.hyper_parameters = hyper_parameters_dict

    def set_best_hyper_parameters(self):
        raise Exception('Need to implement this.')

    def fit(self, X, y):
        self.model = SVC(**self.hyper_parameters)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
