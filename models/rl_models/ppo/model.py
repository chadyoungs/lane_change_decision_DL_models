"""Actor-Critic networks for PPO.

The actor outputs a categorical distribution over the 3 lane-change actions,
and the critic outputs a scalar state-value estimate.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """Stochastic policy (actor) that outputs action logits.

    Parameters
    ----------
    state_dim  : int   Input feature dimension.
    n_actions  : int   Number of discrete actions.
    hidden_dim : int   Width of each hidden layer.
    """

    def __init__(self, state_dim=16, n_actions=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        """Return action logits."""
        return self.net(x)

    def get_dist(self, x):
        """Return a Categorical distribution over actions."""
        return Categorical(logits=self.forward(x))

    def act(self, x):
        """Sample an action and return (action, log_prob)."""
        dist     = self.get_dist(x)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ValueNet(nn.Module):
    """State-value function (critic).

    Parameters
    ----------
    state_dim  : int   Input feature dimension.
    hidden_dim : int   Width of each hidden layer.
    """

    def __init__(self, state_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
