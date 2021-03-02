from .model import DQN, Q_learning, hidden_unit
from .replay_buffer import ReplayBuffer
import copy
import torch
import torch.optim as optim
import random
import numpy as np


class FullControlAgentMaster:
    def __init__(self, trained_model_filename='trained_model.txt', num_actions=25):
        self.num_actions = num_actions
        self.trained_model_filename = trained_model_filename
        self.input_size = 0
        self.model = None
        self.replay_buffer = ReplayBuffer()
        self.optimizer = None
        self.criterion = None
        self.batch_size = 128
        self.gamma = 0.9
        self.epsilon = 1
        self.networks_initialised = False
        self.model_saved = False
        self.model_loaded = False

        self.last_action = 0

        self.update_frequency = 16
        self.steps_since_update = 0

        self.greedy_step_cnt = 0
        self.losses = []

    def init_networks(self, input_size):
        if not self.networks_initialised:
            self.input_size = input_size
            self.model = DQN(input_size, num_actions=self.num_actions)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            self.criterion = torch.nn.MSELoss()
            self.networks_initialised = True

    def save_model(self):
        if not self.model_saved:
            print("saving model...")
            torch.save(self.model.state_dict(), self.trained_model_filename)
            self.model_saved = True

    def load_model(self):
        if not self.model_loaded:
            print("loading model...")
            self.model = DQN(self.input_size, num_actions=self.num_actions)
            self.model.load_state_dict(torch.load(self.trained_model_filename))
            self.model.eval()
            self.model_loaded = True

    def act(self, state):
        state = torch.tensor(state).type(torch.IntTensor)
        q_all_values = self.model.forward(state)
        # print("state: ", state)
        # print("vals: ", q_all_values)
        best_action = q_all_values.max(0)[1].item()
        self.last_action = best_action
        return best_action

    def act_epsilon_greedy(self, state, epsilon=0.1):
        if random.uniform(0, 1) <= epsilon:
            action = random.choices(np.arange(self.num_actions), k=1)[0]
            self.last_action = action
            return action
        else:
            state = torch.tensor(state).type(torch.IntTensor)
            q_all_values = self.model.forward(state)

            self.greedy_step_cnt += 1
            # if self.greedy_step_cnt % 100 == 0:
            # print("state: ", state)
            # print("q vals: ", q_all_values)

            best_action = q_all_values.max(0)[1].item()
            self.last_action = best_action
            return best_action

    def improve_network(self):

        self.steps_since_update += 1
        if self.steps_since_update % self.update_frequency != 0:
            # no need to update
            return

        self.steps_since_update = 0

        # print("improve")
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states).type(torch.LongTensor)
        actions = torch.tensor(actions).type(torch.LongTensor).view(-1, 1)
        rewards = torch.tensor(rewards).type(torch.FloatTensor)
        next_states = torch.tensor(next_states).type(torch.LongTensor)
        dones = torch.tensor(dones).type(torch.LongTensor)

        ongoing_states = (dones == 0)

        q_state_values = self.model(states)
        q_state_action_values = q_state_values.gather(1, actions)

        with torch.no_grad():
            next_q_state_values = self.model(next_states)
            # print(".........next_states", next_states[0])
            # print("--------- values", next_q_state_values[0])

        # print("next_q_state_values: ", next_q_state_values)
        max_q = next_q_state_values.max(1)[0]

        # print("states: ", states)
        # print("actions: ", actions)
        # print("rewards: ", rewards)
        # print("next states: ", next_states)
        # print("dones: ", dones)

        y = rewards
        y[ongoing_states] += self.gamma * max_q[ongoing_states]

        # print("max_q: ", max_q)

        # print(y)

        y = y.view(-1, 1)
        # print("q_state_action: ", q_state_action_values)
        # print("y: ", y)

        loss = self.criterion(q_state_action_values, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())

        # print("last loss: ", self.losses[-1])

        """for p in self.model.parameters():
            p.grad.data.clamp_(-1, 1)"""
        self.optimizer.step()

    def collect_data(self, state, action, reward, next_state, done=0):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.get_size() >= self.batch_size:
            self.improve_network()
