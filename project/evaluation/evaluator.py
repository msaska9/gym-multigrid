import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

training = False

"""
agent type can be:
'random'
'greedy'
'optimal'
'robust'
"""
agent_type = 'optimal'
model_filename = 'trained_model_2_agents_6_size.txt'
simulation_steps = 500


if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v0',
        entry_point='project.envs:CollectGame1Team10x10'
    )

    agents = []

    if agent_type == 'random':
        for i in range(2):
            agents.append(RandomAgent(i))
    elif agent_type == 'greedy':
        for i in range(2):
            agents.append(GreedyAgent(i))
    elif agent_type == 'optimal':
        optimal_agent_master = OptimalAgentMaster(trained_model_filename=model_filename)
        for i in range(2):
            agents.append(OptimalAgent(i, optimal_agent_master))

    env = gym.envs.make('multigrid-collect-1-team-v0', agent_players=agents, number_of_balls=1, is_training=training)
    env.start_simulation()

    visual_mode = 'no-human' if training else 'no-human'
    clock_speed = 0.001 if training else 0.001

    rewards = []

    for i in range(simulation_steps):
        env.render(mode=visual_mode, highlight=False)
        time.sleep(clock_speed)
        env.simulate_round()
        rewards.append(env.get_rewards()[0])
    env.terminate()
    rewards = np.array(rewards)
    print("total reward:", np.sum(rewards))
