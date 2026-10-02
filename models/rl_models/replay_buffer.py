"""Uniform experience replay buffer shared by DQN variants."""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Circular buffer storing (s, a, r, s', done) transitions.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    seed : int, optional
        Random seed for reproducible sampling.
    """

    def __init__(self, capacity=100_000, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.rng    = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random mini-batch.

        Returns
        -------
        states      : np.ndarray  (batch_size, state_dim)
        actions     : np.ndarray  (batch_size,)
        rewards     : np.ndarray  (batch_size,)
        next_states : np.ndarray  (batch_size, state_dim)
        dones       : np.ndarray  (batch_size,) bool
        """
        batch      = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)
