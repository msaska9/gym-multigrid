from ..agent import Agent
import numpy as np
from project.agents.agent import distance_from_ball
from gym_multigrid.multigrid import Actions


class RobustAgent(Agent):
    def __init__(self, agent_id, master_agent, env_type="gym-multigrid"):
        super().__init__(agent_id, agent_type=0, env_type=env_type)
        self.last_observation = None
        self.last_action = None
        self.master_agent = master_agent
        self.rounds = 0

    def process_observation(self, obs, round_id):

        self.observation = obs

        if self.env_type == "my-multigrid":
            return np.array(obs[1] + obs[2])

        pos_x, pos_y = self.get_my_position()
        balls_x, balls_y = self.get_all_ball_positions()
        agents_x, agents_y = self.get_other_agent_positions()
        direction = obs[pos_x][pos_y][1]
        direction_other = obs[agents_x[0]][agents_y[0]][1]

        # features = np.array([pos_x, pos_y, direction, balls_x[0], balls_y[0], agents_x[0], agents_y[0]])

        features = np.array([pos_x, pos_y, direction, agents_x[0], agents_y[0], direction_other, balls_x[0], balls_y[0]])

        return features

    def start_simulation(self, observation, rounds):
        observation = self.process_observation(observation, rounds)
        self.master_agent.init_networks(len(observation))
        if not self.is_training:
            self.master_agent.load_model()

    def next_action(self, observation, reward, round_id):

        pos_x, pos_y = 0, 0
        direction = 0
        ball_x, ball_y = 0, 0

        if self.env_type == "gym-multigrid":
            pos_x, pos_y = self.get_my_position()
            direction = self.observation[pos_x][pos_y][1]
            balls_x, balls_y = self.get_all_ball_positions()
            ball_x, ball_y = balls_x[0], balls_y[0]
        else:
            pos_x, pos_y = self.observation[1][0], self.observation[1][1]
            direction = self.observation[1][2]
            ball_x, ball_y = self.observation[2][0], self.observation[2][1]

        if reward == 1.0:
            if self.last_action == Actions.pickup and \
                    distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y) == 1:
                reward = 1.0
            else:
                # other agent picked up the ball
                # print("other agent picked the ball")
                reward = 0.5

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
