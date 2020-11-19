import gym
import time
from gym.envs.registration import register
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent

if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v0',
        entry_point='project.envs:CollectGame1Team10x10'
    )

    agents = []

    for i in range(6):
        if i < 3:
            agents.append(RandomAgent(i))
        else:
            agents.append(GreedyAgent(i))

    env = gym.envs.make('multigrid-collect-1-team-v0', agent_players=agents, number_of_balls=8)
    env.start_simulation()
    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=False)
        time.sleep(0.1)
        env.simulate_round()
