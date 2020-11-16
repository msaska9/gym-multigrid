from gym_multigrid.envs.collect_game import CollectGameEnv
from project.agents.agent import Agent
import numpy as np


class CollectGame1Team(CollectGameEnv):
    def __init__(
            self,
            size=10,
            num_balls=[],
            agents_index=[],
            balls_index=[],
            balls_reward=[],
            agent_players=[]
    ):
        self.agent_players = []
        self.num_agents = len(agents_index)
        super().__init__(
            size=size,
            num_balls=num_balls,
            agents_index=agents_index,
            balls_index=balls_index,
            balls_reward=balls_reward)
        self.last_observations = None
        self.last_rewards = None
        self.agent_players = agent_players

    def step(self, actions):
        obs, rewards, done, info = super().step(actions)
        obs = np.array(obs)

        agent_positions = []

        extracted_obs = obs[:, :, :, [0, 5]]

        for index, observation in enumerate(extracted_obs):
            (x, y, _) = np.where(observation == [0, 1])
            agent_positions.append([x[0], y[0]])

        for agent_index, [pos_x, pos_y] in enumerate(agent_positions):
            extracted_obs[:, pos_x, pos_y, 1] = agent_index

        return extracted_obs, rewards, done, info

    def simulate_round(self):
        actions = []
        for agent_index, agent in enumerate(self.agent_players):
            obs = self.last_observations[agent_index]
            reward = self.last_rewards[agent_index]
            actions.append(agent.next_action(obs, reward))
        self.last_observations, self.last_rewards, done, info = self.step(actions)

    def reset(self):
        self.last_observations = super().reset()
        self.last_rewards = [0] * len(self.agent_players)


class CollectGame1Team3Agents10x10(CollectGame1Team):
    def __init__(self, agent_players):
        super().__init__(size=10,
                         num_balls=[10],
                         agents_index=[0]*len(agent_players),
                         balls_index=[0],
                         balls_reward=[1],
                         agent_players=agent_players)
