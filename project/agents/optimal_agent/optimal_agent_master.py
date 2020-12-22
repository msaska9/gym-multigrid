from .model import DQN, Q_learning, hidden_unit
from .replay_buffer import ReplayBuffer
import copy
import torch
import torch.optim as optim
import random
import numpy as np

class OptimalAgentMaster:
    def __init__(self, num_actions=5):
        self.num_actions = num_actions
        self.input_size = 0
        self.model = None
        self.replay_buffer = ReplayBuffer()
        self.optimizer = None
        self.criterion = None
        self.batch_size = 40
        self.gamma = 0.9
        self.epsilon = 1
        self.networks_initialised = False

        self.greedy_step_cnt = 0
        self.losses = []

    def init_networks(self, input_size):
        if not self.networks_initialised:
            self.input_size = input_size
            # self.model = Q_learning(input_size, [150, 150], self.num_actions, hidden_unit)
            self.model = DQN(input_size)
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-2)
            self.criterion = torch.nn.MSELoss()
            self.networks_initialised = True

    def act_epsilon_greedy(self, state, epsilon=0.8):
        if random.uniform(0, 1) <= epsilon:
            action = random.choices(np.arange(self.num_actions), weights=[0.1, 0.2, 0.2, 0.4, 0.1], k=1)[0]
            return action
        else:
            state = torch.tensor(state).type(torch.IntTensor)
            q_all_values = self.model.forward(state)

            self.greedy_step_cnt += 1
            #if self.greedy_step_cnt % 100 == 0:
                #print("state: ", state)
                #print("q vals: ", q_all_values)

            best_action = q_all_values.max(0)[1].item()
            return best_action

    def improve_network(self):
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
            print(".........next_states", next_states[0])
            print("--------- values", next_q_state_values[0])

        #print("next_q_state_values: ", next_q_state_values)
        max_q = next_q_state_values.max(1)[0]

        #print("states: ", states)
        #print("actions: ", actions)
        #print("rewards: ", rewards)
        #print("next states: ", next_states)
        #print("dones: ", dones)


        y = rewards
        y[ongoing_states] += self.gamma * max_q[ongoing_states]

        #print("max_q: ", max_q)


        # print(y)



        y = y.view(-1, 1)
        #print("q_state_action: ", q_state_action_values)
        #print("y: ", y)

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