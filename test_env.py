import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='collect', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    else:
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=False)
        time.sleep(1.0)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        for a in ac:
            print("action: ", a)


        obs, reward, done, _ = env.step(ac)

        print(obs)

        print(reward)

        if done:
            break

if __name__ == "__main__":
    main()