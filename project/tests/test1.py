import gym
import time
from gym.envs.registration import register
from project.agents.agent import RandomAgent


if __name__ == '__main__':
    register(
        id='multigrid-collect-1-team-v0',
        entry_point='project.envs:CollectGame1Team10x10'
    )

    agents = [RandomAgent(i) for i in range(3)]

    env = gym.envs.make('multigrid-collect-1-team-v0', agent_players=agents, number_of_balls=40)
    env.reset()
    nb_agents = len(env.agents)



    while True:
        env.render(mode='human', highlight=False)
        time.sleep(1.0)

        #ac = [env.action_space.sample() for _ in range(nb_agents)]
        #ac[0] = int(input())

        #obs, reward, done, _ = env.step(ac)

        env.simulate_round()

        #if done:
        #    break

