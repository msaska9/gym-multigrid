from .model import DQN
from .replay_buffer import ReplayBuffer
import copy
import torch
import torch.optim as optim
import random
import numpy as np


class OptimalAgentMaster:
    def __init__(self, num_actions=5, target_update_frequency=100):
        self.num_actions = num_actions
        self.target_update_frequency = target_update_frequency
        self.time_since_last_update = 0
        self.input_size = 0
        self.q_net = None
        self.target_q_net = None
        self.replay_buffer = ReplayBuffer()
        self.optimizer = None
        self.batch_size = 20
        self.gamma = 0.9
        self.networks_initialised = False

    def init_networks(self, input_size):
        if not self.networks_initialised:
            self.input_size = input_size
            self.q_net = DQN(input_size, self.num_actions)
            self.target_q_net = copy.deepcopy(self.q_net)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.9)
            self.networks_initialised = True

    def act_epsilon_greedy(self, state, epsilon=0.1):
        if random.uniform(0, 1) <= epsilon:
            action = random.choices(np.arange(self.num_actions), weights=[0.1, 0.2, 0.2, 0.4, 0.1], k=1)[0]
            return action
        else:
            state = torch.tensor(state).type(torch.IntTensor)
            q_all_values = self.q_net.forward(state)
            best_action = q_all_values.max(0)[1].item()
            return best_action

    def improve_network(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states).type(torch.LongTensor)
        actions = torch.tensor(actions).type(torch.LongTensor)
        rewards = torch.tensor(rewards).type(torch.LongTensor)
        next_states = torch.tensor(next_states).type(torch.LongTensor)
        dones = torch.tensor(dones).type(torch.LongTensor)

        q_all_values = self.q_net(states)
        q_values = q_all_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_all_values = self.q_net(next_states)
        _, arg_maximums = torch.max(next_q_all_values, dim=1)
        target_q_values = self.target_q_net(next_states)
        next_q_values = torch.gather(target_q_values, 1, arg_maximums.unsqueeze(1))

        expected_q_values = rewards + self.gamma * next_q_values.squeeze() * (1 - dones)
        loss = (q_values - expected_q_values.data).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_parameter, local_parameter in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_parameter.data.copy_(local_parameter.data)

    def collect_data(self, state, action, reward, next_state, done=0):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.get_size() >= self.batch_size:
            self.improve_network()
        self.time_since_last_update += 1
        if self.time_since_last_update == self.target_update_frequency:
            self.update_target_network()
            self.time_since_last_update = 0
