import time
from project.agents.optimal_agent.full_control_agent import FullControlAgent
from project.agents.optimal_agent.full_control_agent_master import FullControlAgentMaster
from project.my_envs.mymultigrid import MyMultiGrid
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

training = True

if __name__ == '__main__':

    if len(sys.argv) > 1:
        training = True if sys.argv[1] == 'train' else False

    full_control_agent_master = FullControlAgentMaster()

    agents = []
    for i in range(2):
        agents.append(FullControlAgent(i, full_control_agent_master, env_type="my-multigrid"))

    env = MyMultiGrid(size=8, num_balls=1, agent_players=agents, is_training=training)
    env.start_simulation()

    visual_mode = 'no-human' if training else 'human'

    time_0 = time.time()

    for i in range(10 * 1000000):
        if i % 1000 == 0:
            print("training #", i)
            time_1 = time.time()
            print("seconds for 1000 steps: ", time_1 - time_0)
            time_0 = time_1
        env.simulate_round()
    env.terminate()