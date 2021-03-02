import random
import numpy as np
from gym_multigrid.multigrid import World
from gym_multigrid.multigrid import DIR_TO_VEC
from gym_multigrid.multigrid import Actions

# random.seed(42)


class Agent:
    def __init__(self, agent_id, agent_type=0, env_type="gym-multigrid"):
        self.id = agent_id
        self.total_reward = 0
        self.action_probabilities = [0.1, 0.2, 0.2, 0.4, 0.1]
        self.agent_type = agent_type
        self.env_type = env_type
        self.observation = None
        self.is_training = True

    def next_action(self, observation, reward, round_id):
        pass

    def start_simulation(self, observation, rounds):
        pass

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        pass

    def save_models(self):
        pass

    def random_action(self):
        action = random.choices(np.arange(5), weights=self.action_probabilities, k=1)[0]
        return action

    def get_my_position(self):
        width = len(self.observation)
        height = len(self.observation[0])
        for x in range(width):
            for y in range(height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["agent"] and self.observation[x][y][2] == self.id:
                    return x, y
        return -1, -1

    def get_other_agent_positions(self):
        width = len(self.observation)
        height = len(self.observation[0])
        positions_x = []
        positions_y = []
        for x in range(width):
            for y in range(height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["agent"] and self.observation[x][y][2] != self.id:
                    positions_x.append(x)
                    positions_y.append(y)
        return positions_x, positions_y

    def get_number_of_agents(self):
        width = len(self.observation)
        height = len(self.observation[0])
        num_agents = 0
        for x in range(width):
            for y in range(height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["agent"]:
                    num_agents += 1
        return num_agents

    def get_all_agent_positions(self):
        width = len(self.observation)
        height = len(self.observation[0])
        num_agents = self.get_number_of_agents()
        positions_x = [0] * num_agents
        positions_y = [0] * num_agents
        for x in range(width):
            for y in range(height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["agent"]:
                    positions_x[self.observation[x][y][2]] = x
                    positions_y[self.observation[x][y][2]] = y
        return positions_x, positions_y

    def get_all_ball_positions(self):
        width = len(self.observation)
        height = len(self.observation[0])
        positions_x = []
        positions_y = []
        for x in range(width):
            for y in range(height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["ball"]:
                    positions_x.append(x)
                    positions_y.append(y)
        return positions_x, positions_y

    def set_training(self, is_training):
        self.is_training = is_training

"""
actions:

still = 0
left = 1
right = 2
forward = 3
pickup = 4
"""


class RandomAgent(Agent):
    def __init__(self, agent_id, env_type="gym-multigrid"):
        super().__init__(agent_id, agent_type=1, env_type=env_type)

    def start_simulation(self, observation, rounds):
        """ Nothing to be done """

    def next_action(self, observation, reward, round_id):
        #print("random index: ", self.id, " type: ", self.agent_type)
        return self.random_action()

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        """ Nothing to be done """


class GreedyAgent(Agent):
    def __init__(self, agent_id, env_type="gym-multigrid"):
        super().__init__(agent_id, agent_type=2, env_type=env_type)
        self.width = 0
        self.height = 0

    def get_ball_positions(self):
        positions = []
        for x in range(self.width):
            for y in range(self.height):
                if self.observation[x][y][0] == World.OBJECT_TO_IDX["ball"]:
                    positions.append([x, y])
        return positions

    def greedy_action(self):
        if self.env_type == "gym-multigrid":
            pos_x, pos_y = self.get_my_position()
            direction = self.observation[pos_x][pos_y][1]
            ball_positions = self.get_ball_positions()
        else:
            pos_x, pos_y = self.observation[1][0], self.observation[1][1]
            direction = self.observation[1][2]
            ball_positions = []
            i = 0
            while i < len(self.observation[2]):
                ball_positions.append([self.observation[2][i], self.observation[2][i+1]])
                i += 2
            print("observation: ", self.observation[2])
        target_ball_positions = get_closest_balls(pos_x, pos_y, direction, ball_positions)
        print("target: ", target_ball_positions)
        target_ball_position = random.choice(target_ball_positions)
        return move_towards_ball(pos_x, pos_y, direction, target_ball_position[0], target_ball_position[1])

    def start_simulation(self, observation, rounds):
        if self.env_type == "gym-multigrid":
            self.width = len(observation)
            self.height = len(observation[0])
        else:
            self.width = observation[0][0]
            self.height = observation[0][1]

    def next_action(self, observation, reward, round_id):
        self.observation = observation
        #print("greedy index: ", self.id, " type: ", x, " ", y)
        return self.greedy_action()

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        """ Nothing to be done """


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y):
    dx = ball_x - pos_x
    dy = ball_y - pos_y
    turns_x = abs(sign(dx) - DIR_TO_VEC[direction][0])
    turns_y = abs(sign(dy) - DIR_TO_VEC[direction][1])
    return abs(dx) + abs(dy) + max(turns_x, turns_y)


def get_closest_balls(pos_x, pos_y, direction, ball_positions):
    if len(ball_positions) == 0:
        return [[0, 0]]
    best_positions = []
    best_distance = -1
    for index, [x, y] in enumerate(ball_positions):
        current_distance = distance_from_ball(pos_x, pos_y, direction, x, y)
        if best_distance == -1 or current_distance < best_distance:
            best_distance = current_distance
            best_positions = [[x, y]]
        elif current_distance == best_distance:
            best_positions.append([x, y])
    return best_positions


def get_next_state(pos_x, pos_y, direction, action):
    if action == Actions.still:
        return pos_x, pos_y, direction
    if action == Actions.left:
        new_direction = (direction + 3) % 4
        return pos_x, pos_y, new_direction
    if action == Actions.right:
        new_direction = (direction + 1) % 4
        return pos_x, pos_y, new_direction
    if action == Actions.forward:
        return pos_x + DIR_TO_VEC[direction][0], pos_y + DIR_TO_VEC[direction][1], direction


def move_towards_ball(pos_x, pos_y, direction, ball_x, ball_y):
    distance = distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y)
    if distance == 1:
        return Actions.pickup
    best_action = Actions.still
    best_next_distance = distance
    for action in [Actions.left, Actions.right, Actions.forward]:
        next_x, next_y, next_direction = get_next_state(pos_x, pos_y, direction, action)
        current_next_distance = distance_from_ball(next_x, next_y, next_direction, ball_x, ball_y)
        if current_next_distance < best_next_distance:
            best_next_distance = current_next_distance
            best_action = action
    return best_action
