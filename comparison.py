# @Time    : 2023/4/23 10:12
# @Author  : ygd
# @FileName: comparison.py
# @Software: PyCharm

from DQN import *
from Dueling_DQN import *
from Double_DQN import *
from D3QN import *

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

DQN_replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
DQN_return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            DQN_episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                DQN_replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                DQN_episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if DQN_replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = DQN_replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            DQN_return_list.append(DQN_episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(DQN_return_list[-10:])
                })
            pbar.update(1)
DQN_episodes_list = list(range(len(DQN_return_list)))
DQN_mv_return = rl_utils.moving_average(DQN_return_list, 9)


Double_DQN_replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Double_DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                   target_update, device)
Double_DQN_return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            Double_DQN_episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                Double_DQN_replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                Double_DQN_episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if Double_DQN_replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = Double_DQN_replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            Double_DQN_return_list.append(Double_DQN_episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(Double_DQN_return_list[-10:])
                })
            pbar.update(1)
Double_DQN_episodes_list = list(range(len(Double_DQN_return_list)))
Double_DQN_mv_return = rl_utils.moving_average(Double_DQN_return_list, 9)


Dueling_replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Dueling_DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                    target_update, device)
Dueling_return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            Dueling_episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                Dueling_replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                Dueling_episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if Dueling_replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = Dueling_replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            Dueling_return_list.append(Dueling_episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(Dueling_return_list[-10:])
                })
            pbar.update(1)
Dueling_episodes_list = list(range(len(Dueling_return_list)))
Dueling_mv_return = rl_utils.moving_average(Dueling_return_list, 9)


D3QN_replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = D3QN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
             target_update, device)
D3QN_return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            D3QN_episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                Dueling_replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                D3QN_episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if Dueling_replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = Dueling_replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            D3QN_return_list.append(D3QN_episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(D3QN_return_list[-10:])
                })
            pbar.update(1)
D3QN_episodes_list = list(range(len(D3QN_return_list)))
D3QN_mv_return = rl_utils.moving_average(D3QN_return_list, 9)
plt.plot(Dueling_episodes_list, Dueling_mv_return, label='Dueling_DQN')
plt.plot(DQN_episodes_list, DQN_mv_return, label='DQN')
plt.plot(D3QN_episodes_list, D3QN_mv_return, label='D3QN')
plt.plot(Double_DQN_episodes_list, Double_DQN_mv_return, label='Double_DQN')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.legend()
plt.show()
