import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def cloning(model,envname,render=False,max_timesteps=None,num_rollouts=20):

    print ("****************************************************")
    print("returnobs is starting.............................")
    print ("****************************************************")

    with tf.Session():
        tf_util.initialize()

        # import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            # print('iterations:', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = list(model.predict(obs[None,:]))
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

        return np.array(observations)



