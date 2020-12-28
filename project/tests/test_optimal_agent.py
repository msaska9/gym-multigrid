import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v0',
        entry_point='project.envs:CollectGame1Team10x10'
    )

    optimal_agent_master = OptimalAgentMaster()

    agents = []
    for i in range(1):
        agents.append(OptimalAgent(i, optimal_agent_master))

    env = gym.envs.make('multigrid-collect-1-team-v0', agent_players=agents, number_of_balls=1)
    env.start_simulation()
    nb_agents = len(env.agents)

    for i in range(10000):
        env.render(mode='no-human', highlight=False)
        time.sleep(0.0001)
        env.simulate_round()
    env.terminate()