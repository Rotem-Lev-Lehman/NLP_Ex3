from base_classifier import BaseClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self):
        super().__init__('Logistic Regression')
        self.hyper_parameters = {}  # dictionary of the chosen hyper-parameters
        self.model = None

    def get_hyper_parameters_grid(self):
        grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l2']}  # l1 lasso l2 ridge
        return grid

    def set_hyper_parameters(self, hyper_parameters_dict):
        self.hyper_parameters = hyper_parameters_dict

    def set_best_hyper_parameters(self):
        raise Exception('Need to implement this.')

    def fit(self, X, y):
        self.model = LogisticRegression(**self.hyper_parameters, max_iter=10000)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
