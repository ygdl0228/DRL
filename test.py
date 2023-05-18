# @Time    : 2023/4/6 22:01
# @Author  : ygd
# @FileName: test.py
# @Software: PyCharm
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from torch import nn
import math
from DQN_noisynet import NoisyReplayBuffer
from DQN_noisynet import NoisyDQN
from DQN import DQNReplayBuffer
from DQN import DQN

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
Noisyreplay_buffer = NoisyReplayBuffer(buffer_size)
Noisyagent = NoisyDQN(state_dim, action_dim, lr, gamma, target_update, device)
Noisyreturn_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            Noisyepisode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = Noisyagent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                Noisyreplay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                Noisyepisode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if Noisyreplay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = Noisyreplay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    Noisyagent.update(transition_dict)
            Noisyreturn_list.append(Noisyepisode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(Noisyreturn_list[-10:])
                })
            pbar.update(1)
Noisyepisodes_list = list(range(len(Noisyreturn_list)))
Noisymv_return = rl_utils.moving_average(Noisyreturn_list, 9)
epsilon=0.01
replay_buffer = DQNReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
DQN_return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            DQN_return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(DQN_return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(DQN_return_list)))
DQN_mv_return = rl_utils.moving_average(DQN_return_list, 9)

plt.plot(Noisyepisodes_list, Noisymv_return, label='DQN')
plt.plot(episodes_list, DQN_mv_return, label='NoisyDQN')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.legend()
plt.title('DQN on {}'.format(env_name))
plt.show()
