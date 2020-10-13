import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(
        self, state_size, action_size, hidden_dim, nonlin=F.relu
    ):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param hidden_dim: dimension of hidden layers
        :param nonlin: nonlinearity to use
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
        self.nonlin = nonlin

    def forward(self, state):
        """
        Compute forward pass over QNetwork
        :param state: state representation for input state
        :return: forward pass result
        """
        x = self.nonlin(self.fc1(state))
        x = self.nonlin(self.fc2(x))
        return self.fc3(x)
