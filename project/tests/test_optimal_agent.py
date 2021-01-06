import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

training = False

if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v0',
        entry_point='project.envs:CollectGame1Team10x10'
    )

    optimal_agent_master = OptimalAgentMaster()

    agents = []
    for i in range(1):
        agents.append(OptimalAgent(i, optimal_agent_master))

    env = gym.envs.make('multigrid-collect-1-team-v0', agent_players=agents, number_of_balls=1, is_training=training)
    env.start_simulation()
    nb_agents = len(env.agents)

    visual_mode = 'no-human' if training else 'human'
    clock_speed = 0.001 if training else 0.5

    for i in range(20000):
        env.render(mode=visual_mode, highlight=False)
        time.sleep(clock_speed)
        env.simulate_round()
    env.terminate()