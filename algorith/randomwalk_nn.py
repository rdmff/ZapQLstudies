#!/usr/bin/env python3
"""
Random Walk baseline with a neural network (untrained) on CartPole.

This script evaluates a policy that selects actions uniformly at random,
but it creates a neural network with the same architecture as used in
Zap Q‑learning (default [20,12,18]) for consistency. The network is never
updated; it exists only to match the naming convention.

Results are saved in a directory like sdata/randomwalk_nn_20x12x18/.
"""

import os
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================== Configuration ==========================
ENV_NAME = 'CartPole-v1'
NUM_SEEDS = 10                     # number of independent runs
EVAL_INTERVAL = 50                 # evaluate every N episodes (for x-axis)
NUM_EVAL_EPISODES = 20             # episodes per evaluation point
MAX_EPISODE_STEPS = 1000           # max steps per episode
HIDDEN_UNITS = [30, 24, 16]        # network architecture (same as paper Figure 1)
RESULT_BASE_DIR = 'sdata'          # parent directory for results
# ===================================================================

# Build result directory name including architecture
ARCH_STR = 'x'.join(map(str, HIDDEN_UNITS))
RESULT_DIR = os.path.join(RESULT_BASE_DIR, f'randomwalk_nn_{ARCH_STR}')
os.makedirs(RESULT_DIR, exist_ok=True)

# Network definition (identical to NET in zapNN.py)
class NET(nn.Module):
    def __init__(self, num_input, hidden_units):
        super(NET, self).__init__()
        self.hidden = nn.Linear(num_input, hidden_units[0])
        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.qvalue = nn.Linear(hidden_units[2], 1)

    def forward(self, s, a):
        # Process state and action
        if isinstance(s, np.ndarray):
            s_flat = s.flatten()
        elif isinstance(s, torch.Tensor):
            s_flat = s.detach().cpu().numpy().flatten()
        else:
            s_flat = np.array(s).flatten()
        a_val = float(a) if isinstance(a, (int, float)) else a.item() if isinstance(a, torch.Tensor) else a
        input_data = np.append(s_flat, a_val)
        x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        x = F.leaky_relu(self.hidden(x))
        x = F.leaky_relu(self.hidden2(x))
        x = F.leaky_relu(self.hidden3(x))
        return self.qvalue(x)

def random_action_episode(env, net, render=False):
    """
    Run one episode with uniformly random actions.
    The net is ignored (present only for consistency).
    """
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 4:      # older API
            obs_next, reward, done, info = step_result
        else:                          # new API (v0.26+)
            obs_next, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        total_reward += reward
        obs = obs_next
    return total_reward, done

def evaluate_random_policy(env, net, num_episodes, render=False):
    """
    Evaluate the random policy over a number of episodes.
    Returns mean steps per episode.
    """
    steps = []
    for _ in range(num_episodes):
        episode_steps, _ = random_action_episode(env, net, render=render)
        steps.append(episode_steps)
    return np.mean(steps)

def run_seed(seed, env, net, eval_interval, num_eval_episodes):
    """
    Run a single seed: evaluate random policy at regular intervals.
    Returns list of mean steps at each evaluation point.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    # We simulate training episodes (even though nothing is learned)
    # to produce a curve over episodes, as in the paper.
    total_episodes = eval_interval * 20   # arbitrary, enough for a smooth curve
    eval_points = []
    for episode in range(eval_interval, total_episodes + 1, eval_interval):
        mean_steps = evaluate_random_policy(env, net, num_eval_episodes)
        eval_points.append(mean_steps)
    return eval_points

def main():
    # Create environment and set max steps
    env = gym.make(ENV_NAME)
    env._max_episode_steps = MAX_EPISODE_STEPS

    all_results = []

    for seed in range(NUM_SEEDS):
        print(f"Running seed {seed+1}/{NUM_SEEDS}...")
        # Create a new network for each seed (random initialization)
        net = NET(num_input=5, hidden_units=HIDDEN_UNITS)
        eval_points = run_seed(seed, env, net, EVAL_INTERVAL, NUM_EVAL_EPISODES)
        all_results.append(eval_points)

    env.close()

    # Convert to numpy
    all_results = np.array(all_results)   # (NUM_SEEDS, num_eval_points)
    mean_steps = np.mean(all_results, axis=0)
    std_steps = np.std(all_results, axis=0)

    # X-axis: episodes at evaluation points
    x = np.arange(1, len(mean_steps)+1) * EVAL_INTERVAL

    # Save results as a .npz file
    np.savez(os.path.join(RESULT_DIR, f'randomwalk_nn_{ARCH_STR}.npz'),
             x=x, mean=mean_steps, std=std_steps, all_results=all_results,
             config={'env': ENV_NAME, 'seeds': NUM_SEEDS,
                     'hidden_units': HIDDEN_UNITS,
                     'eval_interval': EVAL_INTERVAL,
                     'eval_episodes': NUM_EVAL_EPISODES,
                     'max_steps': MAX_EPISODE_STEPS})

    # Also save a simple text file for easy reading
    with open(os.path.join(RESULT_DIR, f'randomwalk_nn_{ARCH_STR}.txt'), 'w') as f:
        f.write(f"Random Walk baseline (uniform actions) with neural network {ARCH_STR}\n")
        f.write(f"Seeds: {NUM_SEEDS}\n")
        f.write(f"Evaluation every {EVAL_INTERVAL} episodes, averaged over {NUM_EVAL_EPISODES} episodes\n\n")
        f.write("Episodes\tMean steps\tStd dev\n")
        for i, ep in enumerate(x):
            f.write(f"{int(ep)}\t{mean_steps[i]:.2f}\t{std_steps[i]:.2f}\n")

    print(f"\nResults saved in {RESULT_DIR}")
    print(f"Data file: randomwalk_nn_{ARCH_STR}.npz")
    print(f"Text summary: randomwalk_nn_{ARCH_STR}.txt")

if __name__ == '__main__':
    main()