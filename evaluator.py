from base_classifier import BaseClassifier
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, auc, roc_curve, f1_score
import itertools
import numpy as np


class Evaluator:

    def __init__(self, clf):
        """ Initializes a new evaluator.

        :param clf: the classifier we want to evaluate
        :type clf: BaseClassifier
        """
        self.clf = clf

    def run_cross_val(self, X, y, cv=10, scoring='accuracy'):
        """ Runs a cross-validation training on the classifier given, and returns the cross-val scores.

        :param X: the training features
        :type X: pd.DataFrame
        :param y: the training target
        :type y: pd.DataFrame
        :param cv: the amount of cross-validations we want to run. Defaults to 10
        :type cv: int
        :param scoring: the scoring method we want to use in the cross-validation run.
                        Possible values: ['accuracy', 'roc_auc', 'f1'] .Defaults to 'accuracy' score
        :type scoring: str
        :return: the cross-val scores
        :rtype: list
        """
        scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            print(f'Running fold number: {i}')
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            self.clf.fit(X_train, y_train)
            if scoring == 'accuracy':
                y_pred = self.clf.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            elif scoring == 'roc_auc':
                y_pred = self.clf.predict_proba(X_val)
                y_pred_pos = y_pred[:, 1]
                fpr, tpr, thresholds = roc_curve(y_val, y_pred_pos)
                score = auc(fpr, tpr)
            else:
                # scoring is f1:
                y_pred = self.clf.predict(X_val)
                score = f1_score(y_val, y_pred)
            scores.append(score)
        return scores

    def optimize_hyper_parameters(self, X, y, cv=3, scoring='accuracy'):
        """ Runs an optimization on the classifier's hyper-parameters, and sets the best parameters in self.clf.

        :param X: the training features
        :type X: pd.DataFrame
        :param y: the training target
        :type y: pd.DataFrame
        :param cv: the amount of cross-validations we want to run. Defaults to 3
        :type cv: int
        :param scoring: the scoring method we want to optimize on.
                        Possible values: ['accuracy', 'roc_auc', 'f1'] .Defaults to 'accuracy' score
        :type scoring: str
        """
        # example of the parameters grid:
        # parameters_options = {'p1': [1, 3, 5],
        #                       'p2': [2, 4, 5],
        #                       'p3': [6, 4, 8]}
        parameters_options = self.clf.get_hyper_parameters_grid()
        all_options = []
        for param in parameters_options.keys():
            options = parameters_options[param]
            all_options.append(options)
        all_possible_combinations = itertools.product(*all_options)

        best_parameters = None
        best_score = None

        for combination in all_possible_combinations:
            current_parameters = {}
            for i, param in enumerate(parameters_options.keys()):
                current_parameters[param] = combination[i]

            print(f'trying out the combination: {current_parameters}')
            self.clf.set_hyper_parameters(current_parameters)
            scores = self.run_cross_val(X=X, y=y, cv=cv, scoring=scoring)
            curr_score = np.mean(scores)
            if best_score is None or best_score < curr_score:
                best_score = curr_score
                best_parameters = current_parameters
            print(f'Current score is: {curr_score}, best score until now is: {best_score}')

        self.clf.set_hyper_parameters(best_parameters)
        return best_score
