from base_classifier import BaseClassifier
from rnn_model import RNNModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import pandas as pd
from embedding_manager import word_to_ix


class RNNClassifier(BaseClassifier):

    def __init__(self):
        super().__init__('RNN')
        self.hyper_parameters = {}  # dictionary of the chosen hyper-parameters
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.sequence_length = None

    def get_hyper_parameters_grid(self):
        grid = {'lr': [0.001, 0.01, 0.1],
                'epochs': [50, 100],
                'n_neurons_fc1': [64, 128, 256],
                'hidden_dim': [64, 128, 256]
                }
        # best params for FF-NN = {'lr': 0.001, 'epochs': 50, 'n_neurons_fc1': 256, 'n_neurons_fc2': 256}
        return grid

    def set_hyper_parameters(self, hyper_parameters_dict):
        self.hyper_parameters = hyper_parameters_dict

    def set_best_hyper_parameters(self):
        raise Exception('Need to implement this.')

    def fit(self, X, y):
        X_tweet_text_tensor, X_other_features_tensor = self.get_X_tensors(X)
        y_tensor = self.convert_to_tensor(y, target=True)
        n_neurons_fc1 = self.hyper_parameters['n_neurons_fc1']
        hidden_dim = self.hyper_parameters['hidden_dim']
        self.model = RNNModel(num_features=len(X.columns), num_class=2, hidden_dim=hidden_dim,
                                n_neurons_fc1=n_neurons_fc1, sequence_length=self.sequence_length)
        self.init_loss_and_optimizer()
        epochs = self.hyper_parameters['epochs']
        n_batches = 20
        for i in range(epochs):
            for i in range(n_batches):
                # Local batches and labels
                local_X1, local_X2, local_y = self.get_batch(X_tweet_text_tensor, X_other_features_tensor, y_tensor,
                                                             n_batches, i)
                self.optimizer.zero_grad()

                y_pred = self.model(local_X1, local_X2)
                loss = self.criterion(y_pred, local_y)
                '''
                aggregated_losses.append(single_loss)

                if i % 25 == 1:
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
                '''
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X_tweet_text_tensor, X_other_features_tensor = self.get_X_tensors(X)
        outputs = self.model(X_tweet_text_tensor, X_other_features_tensor)

        _, predictions = torch.max(outputs, 1)
        return predictions

    def predict_proba(self, X):
        X_tweet_text_tensor, X_other_features_tensor = self.get_X_tensors(X)
        outputs = self.model(X_tweet_text_tensor, X_other_features_tensor)

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

    def get_X_tensors(self, X):
        """ splits the given X df to tweet text indexes for embedding and other extracted features.

        :param X: the df to split
        :type X: pd.DataFrame
        :return: X_tweet_text_tensor, X_other_features_tensor
        :rtype: tuple
        """
        X_tweet_text = X['tweet text']
        X_other_features = X.drop(labels=['tweet text'], axis=1)
        X_tensor_other_features = self.convert_to_tensor(X_other_features, target=False)
        indices_list = []
        for words_list in X_tweet_text.values:
            indices_list.append([word_to_ix[w] for w in words_list])
        X_tensor_tweet_text = torch.LongTensor(indices_list)
        # X_tensor_tweet_text = torch.LongTensor([word_to_ix[w] for w in X_tweet_text.values])
        return X_tensor_tweet_text, X_tensor_other_features

    def get_batch(self, X_tweet_text_tensor, X_other_features_tensor, y_tensor, n_batches, i):
        """ Creates the i'th batch from the given data.

        :param X_tweet_text_tensor: data to get batch from
        :type X_tweet_text_tensor: torch.Tensor
        :param X_other_features_tensor: data to get batch from
        :type X_other_features_tensor: torch.Tensor
        :param y_tensor: data to get batch from
        :type y_tensor: torch.Tensor
        :param n_batches: the amount of total batches we need
        :type n_batches: int
        :param i: the current batch we want to take
        :type i: int
        :return: a tuple of the batched data
        :rtype: tuple
        """
        X1_batch = X_tweet_text_tensor[i * n_batches:(i + 1) * n_batches, ]
        X2_batch = X_other_features_tensor[i * n_batches:(i + 1) * n_batches, ]
        y_batch = y_tensor[i * n_batches:(i + 1) * n_batches, ]
        return X1_batch, X2_batch, y_batch
