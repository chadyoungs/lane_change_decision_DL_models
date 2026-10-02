"""Q-network for DQN and Double DQN."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully-connected Q-network.

    Parameters
    ----------
    state_dim  : int   Input feature dimension (16 for Feature A).
    n_actions  : int   Number of discrete actions (3: LK / LLC / RLC).
    hidden_dim : int   Width of each hidden layer.
    """

    def __init__(self, state_dim=16, n_actions=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)
