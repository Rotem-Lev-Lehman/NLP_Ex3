import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding_manager import embedding_dict


class BasicModel(nn.Module):

    def __init__(self, num_features, num_class, n_neurons_fc1, n_neurons_fc2):
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
        self.embedding_dim = len(embedding_dict[embedding_dict.keys()[0]])
        self.embedding = nn.Embedding(num_embeddings=len(embedding_dict),
                                      embedding_dim=self.embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(...) continue here
        self.input_layer = nn.Linear(num_features + self.embedding_dim, n_neurons_fc1)
        self.fc1 = nn.Linear(n_neurons_fc1, n_neurons_fc2)
        self.fc2 = nn.Linear(n_neurons_fc2, num_class)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, tweet_text, other_features):
        """ implements the forward pass of the model

        :param tweet_text: the input tweet text to pass through the embedding layer
        :type tweet_text: torch.Tensor
        :param other_features: the non-text features to add to the model
        :type other_features: torch.Tensor
        :return: the output tensor (after the pass through the model)
        :rtype: torch.Tensor
        """
        embedded = self.embedding(tweet_text)
        # concatenate the two tensors:
        x = torch.cat((embedded, other_features), dim=1)

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.softmax(x)
        return x