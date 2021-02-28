import random
import numpy as np
from gym_multigrid.window import Window
from .rendering import *


class Actions:
    still = 0
    left = 1
    right = 2
    forward = 3
    pickup = 4


class CellType:
    wall = -3
    empty = -2
    ball = -1


DIR_TO_VEC = [
    # right
    [1, 0],
    # down
    [0, 1],
    # left
    [-1, 0],
    # up
    [0, -1]
]


class MyMultiGrid:

    def __init__(self,
                 size,
                 num_balls,
                 agent_players,
                 is_training):
        self.size = size
        self.num_balls = num_balls
        self.agent_players = agent_players
        self.is_training = is_training
        self.reset()
        self.board = []
        self.agent_positions = []
        self.agent_directions = []
        self.ball_positions = []
        self.round_id = 0
        self.total_num_rounds = 50
        self.last_rewards = []
        self.window = None

    def step_with_agent(self, idx, action):
        if action == Actions.still:
            return 0
        elif action == Actions.left:
            self.agent_directions[idx] = (self.agent_directions[idx] + 3) % 4
            return 0
        elif action == Actions.right:
            self.agent_directions[idx] = (self.agent_directions[idx] + 1) % 4
            return 0
        elif action == Actions.forward:
            new_x = self.agent_positions[idx][0] + DIR_TO_VEC[self.agent_directions[idx]][0]
            new_y = self.agent_positions[idx][1] + DIR_TO_VEC[self.agent_directions[idx]][1]
            if self.board[new_x][new_y] == CellType.empty:
                self.board[new_x][new_y] = idx
                self.board[self.agent_positions[idx][0]][self.agent_positions[idx][1]] = CellType.empty
                self.agent_positions[idx] = [new_x, new_y]
            return 0
        elif action == Actions.pickup:
            target_x = self.agent_positions[idx][0] + DIR_TO_VEC[self.agent_directions[idx]][0]
            target_y = self.agent_positions[idx][1] + DIR_TO_VEC[self.agent_directions[idx]][1]
            if self.board[target_x][target_y] == CellType.ball:
                self.board[target_x][target_y] = CellType.empty
                self.remove_ball(target_x, target_y)
                return 1
            return 0

    def step(self, actions):
        order = np.random.permutation(len(actions))
        rewards = [0] * len(self.agent_players)
        for i in order:
            rewards[i] = self.step_with_agent(i, actions[i])
        self.generate_new_balls()
        return rewards

    def reset(self):

        # -3 : wall
        # -2 : empty
        # -1 : ball
        # >=0 : agent (id)
        self.board = init_board(self.size, self.size)

        self.agent_positions = []
        self.agent_directions = []

        for i in range(len(self.agent_players)):
            x, y = self.get_random_empty_cell()
            self.agent_positions.append([x, y])
            self.agent_directions.append(random.randint(0, 3))
            self.board[x][y] = i

        self.ball_positions = []
        self.generate_new_balls()
        self.round_id = 0
        self.last_rewards = [0] * len(self.agent_players)

    def render(self, mode='human'):
        print("board: ", self.board)

    def start_simulation(self):
        self.reset()
        for agent in self.agent_players:
            agent.set_training(self.is_training)
            obs = self.generate_observation(agent.id)
            agent.start_simulation(obs, self.total_num_rounds)

    def start_new_episode(self):
        reward_sum = np.sum(self.last_rewards)
        for agent in self.agent_players:
            obs = self.generate_observation(agent.id)
            # reward = self.last_rewards[agent.id]
            agent.end_simulation(obs, reward_sum, self.round_id, learn_from=False)
        self.start_simulation()

    def simulate_round(self):
        if self.round_id == self.total_num_rounds:
            self.start_new_episode()
        else:
            actions = []
            reward_sum = np.sum(self.last_rewards)
            for agent in self.agent_players:
                obs = self.generate_observation(agent.id)
                # reward = self.last_rewards[agent.id]
                actions.append(agent.next_action(obs, reward_sum, self.round_id))
            # print("myenv actions: ", actions)
            self.last_rewards = self.step(actions)
            # print("new state: ", self.board)
            self.round_id += 1

    def terminate(self):
        if self.is_training:
            for agent in self.agent_players:
                agent.save_models()

    def get_rewards(self):
        return np.array(self.last_rewards)

    def get_random_empty_cell(self):

        x = random.randint(0, self.size-1)
        y = random.randint(0, self.size-1)

        while self.board[x][y] != CellType.empty:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)

        return x, y

    def generate_new_balls(self):

        while len(self.ball_positions) < self.num_balls:
            x, y = self.get_random_empty_cell()
            self.board[x][y] = CellType.ball
            self.ball_positions.append([x, y])

    def remove_ball(self, x, y):
        ball_index = 0
        for i in range(len(self.ball_positions)):
            if self.ball_positions[0] == x and self.ball_positions[1] == y:
                ball_index = i
                break
        self.ball_positions.pop(ball_index)

    def generate_observation(self, agent_idx):
        sizes = [self.size, self.size]
        ball_positions = []
        agent_positions = [self.agent_positions[agent_idx][0],
               self.agent_positions[agent_idx][1],
               self.agent_directions[agent_idx]]
        for i in range(len(self.agent_players)):
            if i != agent_idx:
                agent_positions.append(self.agent_positions[i][0])
                agent_positions.append(self.agent_positions[i][1])
                agent_positions.append(self.agent_directions[i])
        for ball_pos in self.ball_positions:
            ball_positions.append(ball_pos[0])
            ball_positions.append(ball_pos[1])
        return np.array([sizes, agent_positions, ball_positions])

    def render(self):

        if not self.window:
            self.window = Window('Multigrid')
            self.window.show(block=False)
        img = render_multigrid_as_img(self)
        self.window.show_img(img)


def init_board(n, m):
    # initialise to empty
    board = [[CellType.empty] * m for _ in range(n)]

    for i in range(n):
        # wall
        board[i][0] = CellType.wall
        board[i][-1] = CellType.wall

    for j in range(m):
        # wall
        board[0][j] = CellType.wall
        board[-1][j] = CellType.wall

    return board
