from abc import ABC, abstractmethod
import pandas as pd


class BaseClassifier(ABC):

    def __init__(self, clf_name):
        """ Initializes a new BaseClassifier instance.

        :param clf_name: the classifier's name
        :type clf_name: str
        """
        self.clf_name = clf_name

    @abstractmethod
    def fit(self, X, y):
        """ Fits the algorithm on the train data.

        :param X: train features
        :type X: pd.DataFrame
        :param y: train target
        :type y: pd.DataFrame
        :rtype: None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """ Gets prediction for the given X.

        :param X: test features
        :type X: pd.DataFrame
        :return: a prediction for each example in X
        :rtype: list
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """ Gets prediction probabilities for the given X.

        :param X: test features
        :type X: pd.DataFrame
        :return: the prediction's probability for each example in X
        :rtype: list
        """
        pass

    @abstractmethod
    def set_hyper_parameters(self, hyper_parameters_dict):
        """ Sets the classifier's hyper-parameters.

        :param hyper_parameters_dict: the hyper-parameters to be set.
        :type hyper_parameters_dict: dict
        :rtype: None
        """
        pass

    @abstractmethod
    def get_hyper_parameters_grid(self):
        """ Returns a dictionary with all of the possible hyper-parameters of this classifier, with their possible values.

        :return: dictionary in the following format: {'param_name': [opt1, opt2, opt3, ...], ...}
        :rtype: dict
        """
        pass

    @abstractmethod
    def set_best_hyper_parameters(self):
        """ Sets the hyper-parameters that gave us the best results in this classifier.

        :rtype: None
        """
        pass
