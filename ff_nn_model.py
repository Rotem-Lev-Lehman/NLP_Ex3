import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNNModel(nn.Module):

    def __init__(self, num_features, num_class, n_neurons_fc):
        """ Initializes a BasicModel (FF-NN)

        :param num_features: the amount of non-text features
        :type num_features: int
        :param num_class: the number of different classes
        :type num_class: int
        :param n_neurons_fc: the number of neurons in the second fully connected layer
        :type n_neurons_fc: int
        """
        super().__init__()
        self.input_layer = nn.Linear(num_features, n_neurons_fc)
        self.fc = nn.Linear(n_neurons_fc, num_class)

    def forward(self, x):
        """ implements the forward pass of the model

        :param x: the features we want to use in the model
        :type x: torch.Tensor
        :return: the output tensor (after the pass through the model)
        :rtype: torch.Tensor
        """

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x
