from ..agent import Agent
import numpy as np


class OptimalAgent(Agent):
    def __init__(self, agent_id, master_agent):
        super().__init__(agent_id, agent_type=3)
        self.last_observation = None
        self.last_action = None
        self.master_agent = master_agent
        self.rounds = 0

    def process_observation(self, obs, round_id):
        """
        self.observation = obs
        pos_x, pos_y = self.get_my_position()
        obs = obs[:, :, [0, 1]]
        obs = obs.flatten()
        obs = np.append(obs, pos_x)
        obs = np.append(obs, pos_y)
        obs = np.append(obs, round_id)
        return obs
        """

        #testing with easy features

        self.observation = obs
        pos_x, pos_y = self.get_my_position()
        balls_x, balls_y = self.get_all_ball_positions()
        direction = obs[pos_x][pos_y][1]

        features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], round_id])
        #print(features)
        return features

    def start_simulation(self, observation, rounds):
        observation = self.process_observation(observation, rounds)
        self.master_agent.init_networks(len(observation))

    def next_action(self, observation, reward, round_id):
        observation = self.process_observation(observation, round_id)
        if not (self.last_observation is None):
            self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation)
        self.last_observation = observation
        self.last_action = self.master_agent.act_epsilon_greedy(self.last_observation)

        if reward > 0:
            print(self.rounds, ".round: ", reward)
        elif self.rounds % 100 == 0:
            print("r# ", self.rounds)
        self.rounds += 1
        return self.last_action

    def end_simulation(self, observation, reward, round_id):
        observation = self.process_observation(observation, round_id)
        self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation, done=1)
        self.last_observation = None
        self.last_action = None
