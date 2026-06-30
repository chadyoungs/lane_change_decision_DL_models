"""Dueling Q-network for lane-change decision making.

References
----------
Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning",
ICML 2016.

The network splits into a value stream V(s) and an advantage stream A(s, a),
combined as:

    Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
"""

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Dueling Q-network.

    Parameters
    ----------
    state_dim  : int   Input feature dimension (16 for Feature A).
    n_actions  : int   Number of discrete actions (3: LK / LLC / RLC).
    hidden_dim : int   Width of the shared and stream layers.
    """

    def __init__(self, state_dim=16, n_actions=3, hidden_dim=128):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x):
        feat      = self.feature(x)
        value     = self.value_stream(feat)               # (B, 1)
        advantage = self.advantage_stream(feat)           # (B, n_actions)
        # Combine: subtract mean advantage for identifiability
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
