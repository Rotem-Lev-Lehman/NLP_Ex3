from base_classifier import BaseClassifier
import xgboost as xgb


class XGBoostClassifier(BaseClassifier):

    def __init__(self):
        super().__init__('XGBoost')
        self.hyper_parameters = {}  # dictionary of the chosen hyper-parameters
        self.model = None

    def get_hyper_parameters_grid(self):
        grid = {'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }
        return grid

    def set_hyper_parameters(self, hyper_parameters_dict):
        self.hyper_parameters = hyper_parameters_dict

    def set_best_hyper_parameters(self):
        self.hyper_parameters = {'min_child_weight': 5,
                                 'gamma': 0.5,
                                 'subsample': 0.8,
                                 'colsample_bytree': 0.6,
                                 'max_depth': 5}

    def fit(self, X, y):
        self.model = xgb.XGBClassifier(**self.hyper_parameters)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
