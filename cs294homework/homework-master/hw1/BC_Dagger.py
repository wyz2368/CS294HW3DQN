import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def expert(expert_policy_file,envname,render=False,max_timesteps=None,num_rollouts=20):
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument("--max_timesteps", type=int)
    # parser.add_argument('--num_rollouts', type=int, default=20,
    #                     help='Number of expert roll outs')
    # args = parser.parse_args()


    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        # import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        # print('returns', returns)
        # print('mean return', np.mean(returns))
        # print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        return expert_data

# a = expert('experts/Humanoid-v1.pkl','Humanoid-v1',num_rollouts=20)
# print(np.shape(a['observations']))
# print("***********************************************************")
# print(np.shape(a['actions']))
# print("***********************************************************")

