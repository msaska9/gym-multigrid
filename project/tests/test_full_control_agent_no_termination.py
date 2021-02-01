import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.full_control_agent import FullControlAgent
from project.agents.optimal_agent.full_control_agent_master import FullControlAgentMaster
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

training = False

if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v1',
        entry_point='project.envs:CollectGame1Team6x6NoTermination'
    )

    if len(sys.argv) > 1:
        training = True if sys.argv[1] == 'train' else False

    full_control_agent_master = FullControlAgentMaster()

    agents = []
    for i in range(2):
        agents.append(FullControlAgent(i, full_control_agent_master))

    env = gym.envs.make('multigrid-collect-1-team-v1', agent_players=agents, number_of_balls=1, is_training=training)
    env.start_simulation()
    nb_agents = len(env.agents)

    visual_mode = 'no-human' if training else 'human'
    clock_speed = 0.001 if training else 0.5

    for i in range(1000000):
        if i % 1000 == 0:
            print("training #", i)
        env.render(mode=visual_mode, highlight=False)
        time.sleep(clock_speed)
        env.simulate_round()
    env.terminate()
