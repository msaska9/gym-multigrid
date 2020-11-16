import random
import numpy as np


class Agent:
    def __init__(self, agent_id):
        self.id = id
        self.total_reward = 0

    def next_action(self, observation, reward):
        self.total_reward += reward
        # print("Agent #", self.id, " is acting now.")


"""
actions:

still = 0
left = 1
right = 2
forward = 3
pickup = 4
"""


class RandomAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.action_probabilities = [0.1, 0.2, 0.2, 0.4, 0.1]

    def next_action(self, observation, reward):
        action = random.choices(np.arange(5), weights=self.action_probabilities, k=1)[0]
        return action
