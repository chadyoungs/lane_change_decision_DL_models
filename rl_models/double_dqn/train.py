"""
Double DQN for lane-change decision making.

References
----------
van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016.

The key difference from plain DQN: the online network selects the best next action,
while the target network evaluates it.  This decouples action selection from action
evaluation and reduces overestimation bias.

Usage
-----
    python3 rl_models/double_dqn/train.py
"""

import os
import sys
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PRE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_models.env import LaneChangeEnv
from rl_models.replay_buffer import ReplayBuffer
from rl_models.dqn.model import QNetwork   # reuse the same architecture

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (same as DQN for a fair comparison)
# ──────────────────────────────────────────────────────────────────────────────
NUM_EPISODES     = 3000
BATCH_SIZE       = 64
GAMMA            = 0.99
LR               = 1e-3
BUFFER_CAPACITY  = 100_000
MIN_BUFFER_SIZE  = 1000
TARGET_UPDATE    = 200
HIDDEN_DIM       = 128

EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    data = []
    for i in range(1, 61):
        idx_str  = "{0:02d}".format(i)
        pkl_path = os.path.join(PRE_DIR, "output", f"result{idx_str}_Normal.pickle")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            data.extend(pickle.load(f))
    if not data:
        raise FileNotFoundError(
            "No Normal-feature pickle files found in output/. "
            "Run `python3 calculate/get_time_series_feature.py` first."
        )
    return data


def select_action(q_net, state, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        return int(q_net(s).argmax(dim=1).item())


def train_step(q_net, target_net, optimizer, replay_buffer, loss_fn):
    """Double DQN update: online net selects action, target net evaluates it."""
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states      = torch.FloatTensor(states).to(device)
    actions     = torch.LongTensor(actions).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones       = torch.BoolTensor(dones).to(device)

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: select best action with online net …
        best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        # … evaluate it with target net
        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
        next_q[dones] = 0.0
        targets = rewards + GAMMA * next_q

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def evaluate(q_net, env_data, n_eval=200):
    eval_env = LaneChangeEnv(env_data, seed=99)
    correct  = 0
    for _ in range(n_eval):
        state = eval_env.reset()
        done  = False
        last_action = 0
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                last_action = int(q_net(s).argmax(dim=1).item())
            state, _, done, info = eval_env.step(last_action)
        if last_action == info["label"]:
            correct += 1
    return correct / n_eval


def main():
    os.makedirs(os.path.join(PRE_DIR, "output"), exist_ok=True)
    save_path = os.path.join(PRE_DIR, "output", "double_dqn_best.pth")

    print("Loading data …")
    data = load_data()
    print(f"  {len(data)} episodes loaded.")

    env           = LaneChangeEnv(data, seed=42)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, seed=42)

    q_net      = QNetwork(env.state_dim, env.n_actions, HIDDEN_DIM).to(device)
    target_net = QNetwork(env.state_dim, env.n_actions, HIDDEN_DIM).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    eps         = EPS_START
    total_steps = 0
    best_acc    = 0.0

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done  = False
        ep_loss  = 0.0
        ep_steps = 0

        while not done:
            action = select_action(q_net, state, eps, env.n_actions)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            ep_steps    += 1

            if len(replay_buffer) >= MIN_BUFFER_SIZE:
                loss = train_step(q_net, target_net, optimizer, replay_buffer, loss_fn)
                ep_loss += loss

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        eps = max(EPS_END, eps * EPS_DECAY)

        if episode % 100 == 0:
            acc      = evaluate(q_net, data)
            avg_loss = ep_loss / max(ep_steps, 1)
            print(
                f"Episode {episode:5d} | steps {total_steps:7d} | "
                f"ε {eps:.3f} | loss {avg_loss:.4f} | eval acc {acc:.4f}"
            )
            if acc > best_acc:
                best_acc = acc
                torch.save(q_net.state_dict(), save_path)
                print(f"  ✓ Saved best model (acc={best_acc:.4f}) → {save_path}")

    print(f"\nTraining complete. Best eval accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
