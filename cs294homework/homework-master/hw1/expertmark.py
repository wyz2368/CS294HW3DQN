import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def expertactions(expert_policy_file,observations):

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        actions = []
        for obs in observations:
            action = policy_fn(obs[None, :])
            actions.append(action)
        return np.array(actions)
