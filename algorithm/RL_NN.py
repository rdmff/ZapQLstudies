from __future__ import division
import gym
import math
import os
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import itertools

# ----------------------------
# Neural Network for DQN
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[24, 16, 10]):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.out(x)


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.device = args.device

        self.q_net = QNetwork(state_dim, action_dim, args.hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, args.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(args.buffer_size)

        self.update_counter = 0

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())


# ----------------------------
# Environment helpers
# ----------------------------
def terminal(env):
    """Check if the episode is truly done (pole fallen)."""
    try:
        if hasattr(env, 'unwrapped'):
            x, theta = env.unwrapped.state[0], env.unwrapped.state[2]
        else:
            x, theta = env.state[0], env.state[2]
    except:
        return False

    x_threshold = getattr(env, 'x_threshold', 2.4)
    theta_threshold_radians = getattr(env, 'theta_threshold_radians', 0.20943951023931954)

    done = (x < -x_threshold or x > x_threshold or
            theta < -theta_threshold_radians or theta > theta_threshold_radians)
    return done


def play_dqn(env, agent, episode, train=True, render=False):
    """Run one episode with the agent (training or evaluation)."""
    episode_reward = 0
    obs = env.reset()
    if isinstance(obs, tuple):
        observation, info = obs
    else:
        observation = obs

    done = False
    while not done:
        if render:
            env.render()

        action = agent.act(observation, epsilon=agent.epsilon if train else None)

        step_result = env.step(action)
        if len(step_result) == 4:
            observation_next, reward, done, info = step_result
            truncated = False
        else:
            observation_next, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        episode_reward += reward
        real_done = terminal(env)

        if train:
            agent.learn(observation, action, reward, observation_next, real_done)

        observation = observation_next

    return episode_reward, real_done


def test_dqn(env, agent, render=False):
    """Run evaluation episode following greedy policy."""
    episode_reward = 0
    obs = env.reset()
    if isinstance(obs, tuple):
        observation, info = obs
    else:
        observation = obs

    done = False
    while not done:
        if render:
            env.render()

        action = agent.act(observation, epsilon=0.0)

        step_result = env.step(action)
        if len(step_result) == 4:
            observation_next, reward, done, info = step_result
        else:
            observation_next, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        episode_reward += reward
        observation = observation_next

    return episode_reward, done


# ----------------------------
# Main experiment
# ----------------------------
def OneRun(env, args, exp_idx, render):
    """Run a single DQN experiment."""
    seed = exp_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, args)
    episodes = args.episodes
    eval_interval = args.eval_interval
    eval_episodes = args.eval_episodes

    policy_rewards = []

    for episode in range(episodes):
        episode_reward, _ = play_dqn(env, agent, episode, train=True, render=False)

        if (episode + 1) % eval_interval == 0:
            print(f'Episode {episode+1}/{episodes}: steps = {episode_reward:.0f}')

            # Evaluate
            eval_rewards = []
            for _ in range(eval_episodes):
                reward, _ = test_dqn(env, agent, render)
                eval_rewards.append(reward)
            avg_reward = np.mean(eval_rewards)
            print(f'Test - Average reward: {avg_reward:.2f}')
            policy_rewards.append(avg_reward)

            # Optionally save model checkpoint
            if args.save_models:
                model_path = os.path.join(args.model_dir, f'model_{exp_idx}_{episode+1}.pt')
                agent.save_model(model_path)

    # Save final rewards for this run
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    torch.save(policy_rewards, os.path.join(args.result_dir, f'policy_rewards_{exp_idx}.pt'))

    # Save final model
    if args.save_models:
        final_model_path = os.path.join(args.model_dir, f'model_final_{exp_idx}.pt')
        agent.save_model(final_model_path)


# ----------------------------
# Arguments and main
# ----------------------------
if __name__ == '__main__':
    class Args:
        pass

    args = Args()
    # Training parameters
    args.episodes = 1000
    args.eval_interval = 50
    args.eval_episodes = 20
    args.gamma = 0.85
    args.lr = 0.005
    args.batch_size = 64
    args.buffer_size = 10000
    args.target_update = 100
    args.epsilon_start = 0.2
    args.epsilon_min = 0.01
    args.epsilon_decay = 0.995
    args.hidden_sizes = [30, 24, 16]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories
    args.model_dir = 'sdata/modelNN/'
    args.result_dir = 'sdata/rewards/'
    args.save_models = True

    os.makedirs('sdata', exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Create environment
    env = gym.make('CartPole-v1')
    render = False

    num_experiments = 3
    for i in range(num_experiments):
        print(f'\n--- Running experiment {i+1}/{num_experiments} ---')
        OneRun(env, args, i, render)

    env.close()
    print('\nTraining completed!')