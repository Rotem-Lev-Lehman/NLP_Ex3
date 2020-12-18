from base_classifier import BaseClassifier
from ff_nn_model import FFNNModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import pandas as pd


class FFNNClassifier(BaseClassifier):

    def __init__(self):
        super().__init__('FF-NN')
        self.hyper_parameters = {}  # dictionary of the chosen hyper-parameters
        self.model = None
        self.criterion = None
        self.optimizer = None

    def get_hyper_parameters_grid(self):
        grid = {'lr': [0.0001, 0.001, 0.005, 0.01],
                'epochs': [10, 50, 100],
                'n_batches': [5, 10, 20],
                'n_neurons_fc': [64, 128, 256, 512]
                }
        return grid

    def set_hyper_parameters(self, hyper_parameters_dict):
        self.hyper_parameters = hyper_parameters_dict

    def set_best_hyper_parameters(self):
        self.hyper_parameters = {'lr': 0.001,
                                 'epochs': 10,
                                 'n_batches': 20,
                                 'n_neurons_fc': 512}

    def fit(self, X, y):
        X_tensor = self.convert_to_tensor(X, target=False)
        y_tensor = self.convert_to_tensor(y, target=True)
        n_neurons_fc = self.hyper_parameters['n_neurons_fc']
        self.model = FFNNModel(num_features=len(X.columns), num_class=2, n_neurons_fc=n_neurons_fc)
        self.init_loss_and_optimizer()
        epochs = self.hyper_parameters['epochs']
        n_batches = self.hyper_parameters['n_batches']
        for i in range(epochs):
            for i in range(n_batches):
                # Local batches and labels
                local_X, local_y = self.get_batch(X_tensor, y_tensor, n_batches, i)
                self.optimizer.zero_grad()

                y_pred = self.model(local_X)
                loss = self.criterion(y_pred, local_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X_tensor = self.convert_to_tensor(X, target=False)
        outputs = self.model(X_tensor)

        _, predictions = torch.max(outputs, 1)
        return predictions

    def predict_proba(self, X):
        X_tensor = self.convert_to_tensor(X, target=False)
        outputs = self.model(X_tensor)

        predictions = outputs.detach().numpy()
        return predictions

    def init_loss_and_optimizer(self):
        """ Initializes the loss and optimizer for the current .fit
        """
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.hyper_parameters['lr'])

    def convert_to_tensor(self, df, target=False):
        """ converts the given DataFrame to a tensor.

        :param df: the DataFrame to convert
        :type df: pd.DataFrame
        :param target: indicates whether we are using the features df(False) or the target df(True). Defaults to False
        :type target: bool
        :return: the converted tensor
        :rtype: torch.Tensor
        """
        if target:
            return torch.LongTensor(df.values)
        return torch.FloatTensor(df.values)

    def get_batch(self, X_tensor, y_tensor, n_batches, i):
        """ Creates the i'th batch from the given data.

        :param X_tensor: data to get batch from
        :type X_tensor: torch.Tensor
        :param y_tensor: target to get batch from
        :type y_tensor: torch.Tensor
        :param n_batches: the amount of total batches we need
        :type n_batches: int
        :param i: the current batch we want to take
        :type i: int
        :return: a tuple of the batched data
        :rtype: tuple
        """
        X_batch = X_tensor[i * n_batches:(i + 1) * n_batches, ]
        y_batch = y_tensor[i * n_batches:(i + 1) * n_batches, ]
        return X_batch, y_batch
