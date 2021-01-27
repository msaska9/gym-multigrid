import gym
import time
from gym.envs.registration import register
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.robust_agent import RobustAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent
import numpy as np
import os
import matplotlib.pyplot as plt
from project.plots.heatmap import heatmap, annotate_heatmap

os.environ['KMP_DUPLICATE_LIB_OK']='True'

visual = False

visual_mode = 'human' if visual else 'no-human'
clock_speed = 0.5 if visual else 0.001

"""
agent type can be:
'random'
'greedy'
'optimal'
'robust'
"""

all_agent_types = ['random', 'greedy', 'optimal', 'robust']
optimal_model_filename = 'trained_optimal_long.txt'
robust_model_filename = 'trained_robust_long.txt'
simulation_steps = 2000

optimal_agent_master = OptimalAgentMaster(trained_model_filename=optimal_model_filename)
robust_agent_master = OptimalAgentMaster(trained_model_filename=robust_model_filename)


def add_agent(agent_list, new_agent_type, agent_id):
    if new_agent_type == 'random':
        agent_list.append(RandomAgent(agent_id))
    elif new_agent_type == 'greedy':
        agents.append(GreedyAgent(agent_id))
    elif new_agent_type == 'optimal':
        agents.append(OptimalAgent(agent_id, optimal_agent_master))
    elif new_agent_type == 'robust':
        agents.append(RobustAgent(agent_id, robust_agent_master))


def plot_results(res):

    res = np.array(res)

    fig, ax = plt.subplots()

    im, cbar = heatmap(res, all_agent_types, all_agent_types, ax=ax,
                               cmap="YlGn", cbarlabel="Rewards")
    annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.show()


def print_result_summary(res):
    print("Result summary:")
    for i0, agent_0 in enumerate(all_agent_types):
        for i1, agent_1 in enumerate(all_agent_types):
            print(agent_0, " vs ", agent_1, " : ", res[i0][i1])
    plot_results(res)


if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v1',
        entry_point='project.envs:CollectGame1Team6x6NoTermination'
    )

    n = len(all_agent_types)
    results = [[0.0] * n for i in range(n)]

    for i0, agent_0 in enumerate(all_agent_types):
        for i1, agent_1 in enumerate(all_agent_types):
            agents = []
            add_agent(agents, agent_0, 0)
            add_agent(agents, agent_1, 1)

            env = gym.envs.make('multigrid-collect-1-team-v1', agent_players=agents, number_of_balls=1,
                                is_training=False)
            env.start_simulation()

            rewards = []

            for i in range(simulation_steps):
                env.render(mode=visual_mode, highlight=False)
                time.sleep(clock_speed)
                env.simulate_round()
                rewards.append(env.get_rewards()[0])
            env.terminate()
            rewards = np.array(rewards)
            total_reward = np.sum(rewards)
            print(agent_0, " vs ", agent_1, " ended with a total reward of: ", total_reward)
            results[i0][i1] = total_reward

    print_result_summary(results)
