import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.robust_agent import RobustAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent
from project.agents.human_control_agent import HumanAgent
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

visual = True

"""
agent type can be:
'random'
'greedy'
'optimal'
'robust'
'human'
"""

# agent_type = 'optimal'
agent_types = ['human', 'human']
optimal_model_filename = 'trained_optimal_long.txt'
robust_model_filename = 'trained_robust_no_termination.txt'
episodes = 50
episode_length = 20

optimal_agent_master = None
robust_agent_master = None


if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v1',
        entry_point='project.envs:CollectGame1Team6x6NoTermination'
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
        elif agent_type == 'human':
            agents.append(HumanAgent(i))

    env = gym.envs.make('multigrid-collect-1-team-v1', agent_players=agents, number_of_balls=1, is_training=False)
    env.start_simulation()

    visual_mode = 'human' if visual else 'no-human'
    clock_speed = 0.2 if visual else 0.001

    rewards = []

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
