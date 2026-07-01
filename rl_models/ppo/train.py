"""
PPO (Proximal Policy Optimization) for lane-change decision making.

References
----------
Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.

PPO collects a fixed number of environment steps (a "rollout"), computes
advantages via GAE (Generalised Advantage Estimation), and then performs
multiple mini-batch gradient updates on the clipped surrogate objective.

Usage
-----
    python3 rl_models/ppo/train.py
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
from rl_models.ppo.model import PolicyNet, ValueNet

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
NUM_ITERATIONS    = 300        # number of PPO update iterations
ROLLOUT_STEPS     = 512        # environment steps collected per iteration
MINI_BATCH_SIZE   = 64
PPO_EPOCHS        = 4          # gradient passes over the rollout per iteration
GAMMA             = 0.99       # discount factor
GAE_LAMBDA        = 0.95       # GAE smoothing factor
CLIP_EPS          = 0.2        # PPO clipping ε
VF_COEF           = 0.5        # value function loss coefficient
ENT_COEF          = 0.01       # entropy bonus coefficient
MAX_GRAD_NORM     = 0.5
LR                = 3e-4
HIDDEN_DIM        = 128

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


def collect_rollout(env, policy, value_net, rollout_steps):
    """Collect a fixed number of steps from the environment.

    Returns lists of experience tuples and the final observation for
    bootstrapping the last value.
    """
    states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

    state = env.reset()
    for _ in range(rollout_steps):
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = policy.act(s_t)
            value            = value_net(s_t)

        action_item   = action.item()
        next_state, reward, done, _ = env.step(action_item)

        states.append(state)
        actions.append(action_item)
        log_probs_old.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        state = env.reset() if done else next_state

    # Bootstrap value for the last state
    s_last = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        last_value = value_net(s_last).item()

    return states, actions, log_probs_old, rewards, dones, values, last_value


def compute_gae(rewards, dones, values, last_value, gamma, lam):
    """Compute Generalised Advantage Estimation (GAE) and returns.

    Returns
    -------
    advantages : np.ndarray
    returns    : np.ndarray
    """
    n          = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae        = 0.0
    for t in reversed(range(n)):
        next_val  = last_value if t == n - 1 else values[t + 1]
        mask      = 0.0 if dones[t] else 1.0
        delta     = rewards[t] + gamma * next_val * mask - values[t]
        gae       = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(policy, value_net, optimizer, states, actions, log_probs_old,
               advantages, returns):
    """Perform PPO_EPOCHS gradient updates over the rollout using mini-batches."""
    n = len(states)
    indices = np.arange(n)

    states_t        = torch.FloatTensor(np.array(states)).to(device)
    actions_t       = torch.LongTensor(actions).to(device)
    log_probs_old_t = torch.FloatTensor(log_probs_old).to(device)
    advantages_t    = torch.FloatTensor(advantages).to(device)
    returns_t       = torch.FloatTensor(returns).to(device)

    # Normalise advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    num_updates       = 0

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n, MINI_BATCH_SIZE):
            mb_idx  = indices[start: start + MINI_BATCH_SIZE]
            mb_s    = states_t[mb_idx]
            mb_a    = actions_t[mb_idx]
            mb_lp   = log_probs_old_t[mb_idx]
            mb_adv  = advantages_t[mb_idx]
            mb_ret  = returns_t[mb_idx]

            dist        = policy.get_dist(mb_s)
            new_log_prob = dist.log_prob(mb_a)
            entropy      = dist.entropy().mean()

            ratio        = (new_log_prob - mb_lp).exp()
            surr1        = ratio * mb_adv
            surr2        = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss  = -torch.min(surr1, surr2).mean()

            value_pred   = value_net(mb_s)
            value_loss   = nn.functional.mse_loss(value_pred, mb_ret)

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()),
                MAX_GRAD_NORM
            )
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()
            num_updates       += 1

    return (
        total_policy_loss / num_updates,
        total_value_loss  / num_updates,
        total_entropy     / num_updates,
    )


def evaluate(policy, env_data, n_eval=200):
    eval_env = LaneChangeEnv(env_data, seed=99)
    correct  = 0
    for _ in range(n_eval):
        state = eval_env.reset()
        done  = False
        last_action = 0
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist        = policy.get_dist(s)
                last_action = int(dist.probs.argmax(dim=1).item())
            state, _, done, info = eval_env.step(last_action)
        if last_action == info["label"]:
            correct += 1
    return correct / n_eval


def main():
    os.makedirs(os.path.join(PRE_DIR, "output"), exist_ok=True)
    save_path = os.path.join(PRE_DIR, "output", "ppo_best.pth")

    print("Loading data …")
    data = load_data()
    print(f"  {len(data)} episodes loaded.")

    env        = LaneChangeEnv(data, seed=42)
    policy     = PolicyNet(env.state_dim, env.n_actions, HIDDEN_DIM).to(device)
    value_net  = ValueNet(env.state_dim, HIDDEN_DIM).to(device)

    optimizer  = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=LR
    )

    best_acc = 0.0

    for iteration in range(1, NUM_ITERATIONS + 1):
        states, actions, log_probs_old, rewards, dones, values, last_value = \
            collect_rollout(env, policy, value_net, ROLLOUT_STEPS)

        advantages, returns = compute_gae(
            rewards, dones, values, last_value, GAMMA, GAE_LAMBDA
        )

        pl, vl, ent = ppo_update(
            policy, value_net, optimizer,
            states, actions, log_probs_old, advantages, returns
        )

        if iteration % 10 == 0:
            acc = evaluate(policy, data)
            print(
                f"Iter {iteration:4d} | policy_loss {pl:.4f} | "
                f"value_loss {vl:.4f} | entropy {ent:.4f} | eval acc {acc:.4f}"
            )
            if acc > best_acc:
                best_acc = acc
                torch.save(policy.state_dict(), save_path)
                print(f"  ✓ Saved best policy (acc={best_acc:.4f}) → {save_path}")

    print(f"\nTraining complete. Best eval accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
