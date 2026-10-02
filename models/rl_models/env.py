"""
Lane change decision environment backed by pre-processed highD Normal-feature pickles.

Each episode corresponds to one driving event (lane change or lane keep) recorded in
the highD dataset.  The agent observes the 16-dimensional Feature-A state vector at
every frame (25 fps → 50 frames ≈ 2 seconds) and must decide at each step whether to

    0 – keep lane (LK)
    1 – change to left lane (LLC)
    2 – change to right lane (RLC)

The ground-truth action label attached to the event is used both for terminal reward
shaping and for evaluation.  Because the replay data is offline, the next state always
advances along the recorded trajectory regardless of the agent's choice; this is the
standard data-driven simulation setup used in offline / batch RL for autonomous driving.

State vector (16 dims, Feature A):
    [left_lane_exist, right_lane_exist,
     delta_y, y_velocity, y_acceleration,
     x_velocity, x_acceleration, car_type,
     preceding_ttc, following_ttc,
     left_preceding_ttc, left_alongside_ttc, left_following_ttc,
     right_preceding_ttc, right_alongside_ttc, right_following_ttc]

Reward:
    • Per-step safety shaping: small penalty when any TTC < TTC_THRESHOLD.
    • Per-step comfort shaping: small penalty for large lateral/longitudinal acceleration.
    • Terminal step: +CORRECT_REWARD if agent action == ground-truth label,
                    -WRONG_PENALTY otherwise.
"""

import random
import numpy as np


# Indices inside the 16-dim Feature A tuple
IDX_LEFT_LANE_EXIST   = 0
IDX_RIGHT_LANE_EXIST  = 1
IDX_DELTA_Y           = 2
IDX_Y_VEL             = 3
IDX_Y_ACC             = 4
IDX_X_VEL             = 5
IDX_X_ACC             = 6
IDX_CAR_TYPE          = 7
IDX_TTC_PRECEDING     = 8
IDX_TTC_FOLLOWING     = 9
IDX_TTC_LEFT_PREC     = 10
IDX_TTC_LEFT_ALONG    = 11
IDX_TTC_LEFT_FOLLOW   = 12
IDX_TTC_RIGHT_PREC    = 13
IDX_TTC_RIGHT_ALONG   = 14
IDX_TTC_RIGHT_FOLLOW  = 15

STATE_DIM  = 16
N_ACTIONS  = 3   # LK=0, LLC=1, RLC=2

# Reward hyper-parameters
TTC_THRESHOLD   = 2.0    # seconds — safety boundary
SAFETY_PENALTY  = -0.05  # per-step penalty when TTC < threshold
COMFORT_PENALTY = -0.01  # per-step penalty per unit of abs lateral acceleration
CORRECT_REWARD  = 1.0    # terminal reward for correct prediction
WRONG_PENALTY   = -1.0   # terminal reward for wrong prediction

# TTC columns used for safety shaping (all 8 surrounding TTC features)
TTC_INDICES = [
    IDX_TTC_PRECEDING, IDX_TTC_FOLLOWING,
    IDX_TTC_LEFT_PREC, IDX_TTC_LEFT_ALONG, IDX_TTC_LEFT_FOLLOW,
    IDX_TTC_RIGHT_PREC, IDX_TTC_RIGHT_ALONG, IDX_TTC_RIGHT_FOLLOW,
]


class LaneChangeEnv:
    """Data-driven lane-change decision environment.

    Parameters
    ----------
    data : list of (state_sequence, label)
        Pre-processed episodes loaded from a Normal-feature pickle file.
        ``state_sequence`` is a list of 16-dim tuples; ``label`` ∈ {0, 1, 2}.
    seed : int, optional
        Random seed for episode sampling.
    """

    def __init__(self, data, seed=42):
        self.data   = data
        self.rng    = random.Random(seed)
        self._seq   = None
        self._label = None
        self._t     = 0
        self._T     = 0

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self):
        """Sample a new episode and return the initial state."""
        self._seq, self._label = self.rng.choice(self.data)
        self._T = len(self._seq)
        self._t = 0
        return self._get_obs()

    def step(self, action):
        """Advance one time step.

        Parameters
        ----------
        action : int
            Agent's action ∈ {0, 1, 2}.

        Returns
        -------
        next_obs : np.ndarray, shape (STATE_DIM,)
        reward   : float
        done     : bool
        info     : dict  {'label': ground_truth_action, 't': current_step}
        """
        state  = self._get_obs()
        reward = self._compute_reward(state, action, terminal=(self._t == self._T - 1))

        self._t += 1
        done     = self._t >= self._T
        next_obs = self._get_obs() if not done else np.zeros(STATE_DIM, dtype=np.float32)

        return next_obs, reward, done, {"label": self._label, "t": self._t}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        raw = self._seq[self._t]
        obs = np.array(raw, dtype=np.float32)
        # Clip extreme TTC values (e.g. 999) to a finite range for stability
        obs[TTC_INDICES] = np.clip(obs[TTC_INDICES], 0.0, 30.0)
        return obs

    def _compute_reward(self, state, action, terminal):
        reward = 0.0

        # Safety shaping: penalise steps where any surrounding TTC is very small
        for idx in TTC_INDICES:
            ttc = float(state[idx])
            if 0.0 < ttc < TTC_THRESHOLD:
                reward += SAFETY_PENALTY

        # Comfort shaping: penalise large lateral (lane-change) acceleration
        reward += COMFORT_PENALTY * abs(float(state[IDX_Y_ACC]))

        # Terminal reward
        if terminal:
            if action == self._label:
                reward += CORRECT_REWARD
            else:
                reward += WRONG_PENALTY

        return reward

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self):
        return STATE_DIM

    @property
    def n_actions(self):
        return N_ACTIONS
