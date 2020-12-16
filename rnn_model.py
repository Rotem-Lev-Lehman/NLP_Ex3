import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding_manager import pretrained_weights, vocab


class RNNModel(nn.Module):

    def __init__(self, num_features, num_class, hidden_dim, n_neurons_fc1, sequence_length):
        """ Initializes a BasicModel (FF-NN)

        :param num_features: the amount of non-text features
        :type num_features: int
        :param num_class: the number of different classes
        :type num_class: int
        :param n_neurons_fc1: the number of neurons in the first fully connected layer
        :type n_neurons_fc1: int
        :param n_neurons_fc2: the number of neurons in the second fully connected layer
        :type n_neurons_fc2: int
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = 300 #len(embedding_dict[embedding_dict.keys()[0]])
        # self.embedding = nn.Embedding(len(vocab), self.embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.input_layer = nn.Linear(num_features - 1 + hidden_dim * 2, n_neurons_fc1)
        self.fc1 = nn.Linear(n_neurons_fc1, num_class)

    def forward(self, tweet_text_idx, other_features):
        """ implements the forward pass of the model

        :param tweet_text: the input tweet text to pass through the embedding layer
        :type tweet_text: torch.Tensor
        :param other_features: the non-text features to add to the model
        :type other_features: torch.Tensor
        :return: the output tensor (after the pass through the model)
        :rtype: torch.Tensor
        """
        embedded = self.embedding(tweet_text_idx)

        lstm_out, _ = self.lstm(embedded.view(len(tweet_text_idx), self.sequence_length, -1))
        lstm_features = lstm_out[:, -1, :]

        # concatenate the two tensors:
        x = torch.cat((lstm_features, other_features), dim=1)

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.softmax(x)
        return x
