from gym_multigrid.envs.collect_game import CollectGameEnv
import numpy as np

class CollectGame1Team(CollectGameEnv):
    def __init__(
            self,
            size=10,
            num_balls=[],
            agents_index=[],
            balls_index=[],
            balls_reward=[],
            agent_players=[],
            total_num_rounds=10,
            is_training=True
    ):
        self.agent_players = []
        self.num_agents = len(agents_index)
        self.round_id = 0
        self.total_num_rounds = total_num_rounds
        self.is_training = is_training

        agent_types = [agent.agent_type for agent in agent_players]

        super().__init__(
            size=size,
            num_balls=num_balls,
            agents_index=agents_index,
            balls_index=balls_index,
            balls_reward=balls_reward,
            agent_types=agent_types
        )
        self.last_observations = None
        self.last_rewards = None
        self.agent_players = agent_players
        self.num_balls = num_balls

    def step(self, actions):
        obs, rewards, done, info = super().step(actions)
        return extract_observation(obs), rewards, done, info

    def start_simulation(self):
        observation = self.reset()
        for agent_index, agent in enumerate(self.agent_players):
            agent.set_training(self.is_training)
            agent.start_simulation(observation[agent_index], self.total_num_rounds)

    def simulate_round(self):
        if self.round_id > self.total_num_rounds:
            for agent_index, agent in enumerate(self.agent_players):
                obs = self.last_observations[agent_index]
                reward = self.last_rewards[agent_index]
                agent.end_simulation(obs, reward, self.round_id)
            self.start_simulation()
        else:
            actions = []
            for agent_index, agent in enumerate(self.agent_players):
                obs = self.last_observations[agent_index]
                reward = self.last_rewards[agent_index]
                actions.append(agent.next_action(obs, reward, self.round_id))
            self.last_observations, self.last_rewards, done, info = self.step(actions)
            self.round_id += 1

    def reset(self):
        obs = super().reset()
        self.last_observations = extract_observation(obs)
        self.last_rewards = [0] * len(self.agent_players)
        self.round_id = 0
        return self.last_observations

    def terminate(self):
        if self.is_training:
            for agent in self.agent_players:
                agent.save_models()


class CollectGame1Team10x10(CollectGame1Team):
    def __init__(self, agent_players, number_of_balls, is_training):
        super().__init__(size=5,
                         num_balls=[number_of_balls],
                         agents_index=[0] * len(agent_players),
                         balls_index=[0],
                         balls_reward=[1],
                         agent_players=agent_players,
                         is_training=is_training)


def extract_observation(obs):
    obs = np.array(obs)

    agent_positions = []

    # row 0 gives the object type
    # row 5 gives self / other agent

    is_agent = obs[:, :, :, [5]]

    for index, observation in enumerate(is_agent):
        (x, y, _) = np.where(observation == [1])
        agent_positions.append([x[0], y[0]])
        # print(index, " agent: ", x[0], " ", y[0])

    extracted_obs = obs[:, :, :, [0, 4, 5]]

    for agent_index, [pos_x, pos_y] in enumerate(agent_positions):
        extracted_obs[:, pos_x, pos_y, 2] = agent_index

    return extracted_obs
