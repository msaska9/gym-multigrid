from ..agent import Agent
import numpy as np


class OptimalAgent(Agent):
    def __init__(self, agent_id, master_agent, env_type="gym-multigrid"):
        super().__init__(agent_id, agent_type=3, env_type=env_type)
        self.last_observation = None
        self.last_action = None
        self.master_agent = master_agent
        self.rounds = 0

    def process_observation(self, obs, round_id):




        self.observation = obs

        if self.env_type == "my-multigrid":
            return np.array(obs[1] + obs[2])

        """"# pos_x, pos_y = self.get_my_position()
        # obs = obs[:, :, [0, 1]]
        obs = obs.flatten()
        # obs = np.append(obs, pos_x)
        # obs = np.append(obs, pos_y)
        obs = np.append(obs, round_id)

        print("obs: ", obs)

        return obs"""


        #testing with easy features

        self.observation = obs
        pos_x, pos_y = self.get_my_position()
        balls_x, balls_y = self.get_all_ball_positions()
        agents_x, agents_y = self.get_other_agent_positions()
        direction = obs[pos_x][pos_y][1]
        direction_other = obs[agents_x[0]][agents_y[0]][1]

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], round_id])

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0]])

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], balls_x[1], balls_y[1], round_id])

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], balls_x[1], balls_y[1]])

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], agents_x[0], agents_y[0]])

        features = np.array([pos_x, pos_y, direction, agents_x[0], agents_y[0], direction_other, balls_x[0], balls_y[0]])


        #print(features)
        return features

    def start_simulation(self, observation, rounds):
        observation = self.process_observation(observation, rounds)
        self.master_agent.init_networks(len(observation))
        if not self.is_training:
            self.master_agent.load_model()

    def next_action(self, observation, reward, round_id):

        # reward = 1.0

        observation = self.process_observation(observation, round_id)
        if not (self.last_observation is None) and self.is_training:
            self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation)
        self.last_observation = observation
        if self.is_training:
            self.last_action = self.master_agent.act_epsilon_greedy(self.last_observation)
        else:
            self.last_action = self.master_agent.act(self.last_observation)

        """if reward > 0:
            print(self.rounds, ".round: ", reward)"""
        """if self.rounds % 100 == 0:
            print("r# ", self.rounds)"""
        self.rounds += 1
        return self.last_action

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        observation = self.process_observation(observation, round_id)
        if not (self.last_observation is None) and self.is_training and learn_from:
            self.master_agent.collect_data(self.last_observation, self.last_action, reward, observation, done=1)
        self.last_observation = None
        self.last_action = None

    def save_models(self):
        self.master_agent.save_model()
