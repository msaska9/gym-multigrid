import gym
import time
from project.agents.optimal_agent.optimal_agent import OptimalAgent
from project.agents.optimal_agent.robust_agent import RobustAgent
from project.agents.optimal_agent.optimal_agent_master import OptimalAgentMaster
from project.agents.optimal_agent.full_control_agent_master import FullControlAgentMaster
from project.agents.optimal_agent.full_control_agent import FullControlAgent
from project.my_envs.mymultigrid import MyMultiGrid
from project.agents.agent import RandomAgent
from project.agents.agent import GreedyAgent
import numpy as np
import os
import matplotlib.pyplot as plt
from project.plots.heatmap import heatmap, annotate_heatmap

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

visual = False

visual_mode = 'human' if visual else 'no-human'
clock_speed = 0.5 if visual else 0.001

"""
agent type can be:
'random'
'greedy'
'optimal'
'robust'
'full_control'
"""

all_agent_types = ['random', 'greedy', 'optimal', 'robust']
# all_agent_types = ['random, optimal']
# all_agent_types = ['full_control']
# optimal_model_filename = 'trained_optimal_long.txt'
optimal_model_filename = 'trained_optimal_8_2b.txt'
# robust_model_filename = 'trained_robust_long.txt'
robust_model_filename = 'trained_robust_8_2b.txt'
full_control_model_filename = 'trained_full_control_long.txt'
episodes = 10000
episode_length = 20

optimal_agent_master = OptimalAgentMaster(trained_model_filename=optimal_model_filename)
robust_agent_master = OptimalAgentMaster(trained_model_filename=robust_model_filename)
full_control_agent_master = FullControlAgentMaster(trained_model_filename=full_control_model_filename)


def add_agent(agent_list, new_agent_type, agent_id):
    if new_agent_type == 'random':
        agent_list.append(RandomAgent(agent_id, env_type="my-multigrid"))
    elif new_agent_type == 'greedy':
        agents.append(GreedyAgent(agent_id, env_type="my-multigrid"))
    elif new_agent_type == 'optimal':
        agents.append(OptimalAgent(agent_id, optimal_agent_master, env_type="my-multigrid"))
    elif new_agent_type == 'robust':
        agents.append(RobustAgent(agent_id, robust_agent_master, env_type="my-multigrid"))
    elif new_agent_type == 'full_control':
        agents.append(FullControlAgent(agent_id, full_control_agent_master, env_type="my-multigrid"))


def plot_results(res):
    res = np.array(res)

    fig, ax = plt.subplots()

    im, cbar = heatmap(res, all_agent_types, all_agent_types, ax=ax,
                       cmap="YlGn", cbarlabel="Rewards")
    annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()


def print_result_summary(res):
    print("Result summary:")
    for i0, agent_0 in enumerate(all_agent_types):
        for i1, agent_1 in enumerate(all_agent_types):
            print(agent_0, " vs ", agent_1, " : ", res[i0][i1])
    plot_results(res)


if __name__ == '__main__':

    n = len(all_agent_types)
    results = [[0.0] * n for i in range(n)]

    for i0, agent_0 in enumerate(all_agent_types):
        for i1, agent_1 in enumerate(all_agent_types):

            if i1 < i0:
                continue

            agents = []
            add_agent(agents, agent_0, 0)
            add_agent(agents, agent_1, 1)

            env = MyMultiGrid(size=8, num_balls=2, agent_players=agents, is_training=False)
            env.start_simulation()

            rewards = []

            time_0 = time.time()

            for episode in range(episodes):
                print("Episode # ", episode)
                reward = 0.0
                for step in range(episode_length):
                    if visual:
                        env.render()
                        time.sleep(1.0)
                    env.simulate_round()
                    reward += np.sum(env.get_rewards())
                print("Episode ended with reward: ", reward)
                rewards.append(reward)
                env.start_new_episode()
            env.terminate()
            print("rewards: ", rewards)
            avg_reward = np.average(rewards)
            print(avg_reward)
            print(agent_0, " vs ", agent_1, " ended with a total reward of: ", avg_reward)
            results[i0][i1] = avg_reward
            results[i1][i0] = avg_reward

    print_result_summary(results)
