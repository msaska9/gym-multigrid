import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.robust_agent import RobustAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
from project.agents.optimal_agent.full_control_agent_master import FullControlAgentMaster
from project.agents.optimal_agent.full_control_agent import FullControlAgent
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent
from project.agents.human_control_agent import HumanAgent
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

visual = False

"""
agent type can be:
'random'
'greedy'
'optimal'
'robust'
'full_control'
'human'
"""

# agent_type = 'optimal'
agent_types = ['optimal', 'optimal']
# optimal_model_filename = 'trained_optimal_long.txt'
# optimal_model_filename = 'trained_optimal_myenv_fast.txt'
optimal_model_filename = 'trained_optimal_8.txt'
# robust_model_filename = 'trained_robust_no_termination.txt'
robust_model_filename = 'trained_robust_8.txt'
full_control_model_filename = 'trained_full_control_long.txt'
episodes = 50
episode_length = 20

optimal_agent_master = None
robust_agent_master = None
full_control_agent_master = None


if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v1',
        entry_point='project.envs:CollectGame1Team8x8NoTermination'
    )

    agents = []

    for i, agent_type in enumerate(agent_types):
        if agent_type == 'random':
            agents.append(RandomAgent(i))
        elif agent_type == 'greedy':
            agents.append(GreedyAgent(i))
        elif agent_type == 'optimal':
            if optimal_agent_master is None:
                optimal_agent_master = OptimalAgentMaster(trained_model_filename=optimal_model_filename)
            agents.append(OptimalAgent(i, optimal_agent_master))
        elif agent_type == 'robust':
            if robust_agent_master is None:
                robust_agent_master = OptimalAgentMaster(trained_model_filename=robust_model_filename)
            agents.append(RobustAgent(i, robust_agent_master))
        elif agent_type == 'full_control':
            if full_control_agent_master is None:
                full_control_agent_master = FullControlAgentMaster(trained_model_filename=full_control_model_filename)
            agents.append(FullControlAgent(i, full_control_agent_master))
        elif agent_type == 'human':
            agents.append(HumanAgent(i))

    env = gym.envs.make('multigrid-collect-1-team-v1', agent_players=agents, number_of_balls=1, is_training=False)
    env.start_simulation()

    visual_mode = 'human' if visual else 'no-human'
    clock_speed = 0.2 if visual else 0.001

    rewards = []

    time_0 = time.time()

    for episode in range(episodes):
        print("Episode # ", episode)
        reward = 0.0
        for step in range(episode_length):
            env.render(mode=visual_mode, highlight=False)
            time.sleep(clock_speed)
            env.simulate_round()
            reward += env.get_rewards()[0]
        print("Episode ended with reward: ", reward)
        rewards.append(reward)
        env.start_new_episode()
    env.terminate()
    print("rewards: ", rewards)
    print(np.average(rewards))

    time_1 = time.time()

    print(time_1 - time_0)
