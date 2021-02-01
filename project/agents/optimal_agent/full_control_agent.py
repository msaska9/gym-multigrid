from ..agent import Agent
import numpy as np


class FullControlAgent(Agent):
    def __init__(self, agent_id, master_agent):
        super().__init__(agent_id, agent_type=5)
        self.last_observation = None
        self.last_action = None
        self.master_agent = master_agent
        self.rounds = 0

    def process_observation(self, obs, round_id):

        self.observation = obs
        balls_x, balls_y = self.get_all_ball_positions()
        agents_x, agents_y = self.get_all_agent_positions()
        direction_0 = obs[agents_x[0]][agents_y[0]][1]
        direction_1 = obs[agents_x[1]][agents_y[1]][1]

        features = np.array([agents_x[0], agents_y[0], direction_0, agents_x[1], agents_y[1], direction_1, balls_x[0], balls_y[0]])


        #print(features)
        return features

    def start_simulation(self, observation, rounds):
        observation = self.process_observation(observation, rounds)
        self.master_agent.init_networks(len(observation))
        if not self.is_training:
            self.master_agent.load_model()

    def next_action(self, observation, reward, round_id):

        self.rounds += 1

        if self.id > 0:
            return self.master_agent.last_action // 5

        observation = self.process_observation(observation, round_id)
        if not (self.last_observation is None) and self.is_training:
            self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation)
        self.last_observation = observation
        if self.is_training:
            self.last_action = self.master_agent.act_epsilon_greedy(self.last_observation)
        else:
            self.last_action = self.master_agent.act(self.last_observation)

        return self.master_agent.last_action % 5

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        observation = self.process_observation(observation, round_id)
        if not (self.last_observation is None) and self.is_training and learn_from:
            self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation, done=1)
        self.last_observation = None
        self.last_action = None

    def save_models(self):
        self.master_agent.save_model()
