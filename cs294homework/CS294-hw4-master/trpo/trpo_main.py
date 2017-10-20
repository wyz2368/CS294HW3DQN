from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
from gym import envs, scoreboard
import tempfile
import sys
from collections import namedtuple


tiny = 1e-10


def create_config():
    MyConfig = namedtuple('MyConfig', ['timesteps_per_batch', 'max_pathlength', 'max_kl', 'cg_damping', 'gamma'])
    config = MyConfig(timesteps_per_batch=6000,
                      max_pathlength=200,
                      max_kl=0.01,
                      cg_damping=0.1,
                      gamma=0.99)
    return config


def normal_log_prob(x, mean, log_std, dim):
    """
    x: [batch, dim]
    return: [batch]
    """
    zs = (x - mean) / tf.exp(log_std)
    return - tf.reduce_sum(log_std, axis=1) - \
           0.5 * tf.reduce_sum(tf.square(zs), axis=1) - \
           0.5 * dim * np.log(2 * np.pi)


def normal_kl(old_mean, old_log_std, new_mean, new_log_std):
    """
    mean, log_std: [batch,  dim]
    return: [batch]
    """
    old_std = tf.exp(old_log_std)
    new_std = tf.exp(new_log_std)
    numerator = tf.square(old_mean - new_mean) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2 * tf.square(new_std) + tiny
    return tf.reduce_sum(
        numerator / denominator + new_log_std - old_log_std, axis=1)


def normal_entropy(log_std):
    return tf.reduce_sum(log_std + np.log(np.sqrt(2 * np.pi * np.e)), axis=1)


class TRPOAgent(object):
    def __init__(self, env):
        self.config = create_config()
        self.env = env

        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)

        self.obs_dim = obs_dim = env.observation_space.shape[0]
        self.act_dim = act_dim = env.action_space.shape[0]

        self.session = tf.Session()
        self.end_count = 0
        self.train = True

        # obs = [ current obs, previous obs, previous action ]
        self.obs = obs = tf.placeholder(tf.float32, shape=[None, obs_dim], name="obs")
        self.prev_obs = np.zeros((1, obs_dim), dtype=np.float32)
        self.prev_action = np.zeros((1, act_dim), dtype=np.float32)
        self.action = action = tf.placeholder(tf.float32, shape=[None, act_dim], name="action")
        self.advant = advant = tf.placeholder(tf.float32, shape=[None], name="advant")

        self.oldact_mean = oldact_mean = tf.placeholder(tf.float32, shape=[None, act_dim], name="oldaction_mean")
        self.oldact_logstd = oldact_logstd = tf.placeholder(tf.float32, shape=[None, act_dim], name="oldaction_logstd")

        # Create neural network.
        layer_h1 = tf.nn.relu(dense(obs, 32, 'hidden1', 1.0))
        layer_h2 = tf.nn.relu(dense(layer_h1, 8, 'hidden2', 1.0))
        self.act_mean = act_mean = dense(layer_h2, act_dim, 'action_mean', 0.1)
        self.act_logstd = act_logstd = dense(layer_h2, act_dim, 'action_logstd', 0.1)

        # sample action
        sampled_eps = tf.random_normal(tf.shape(act_mean))
        self.sampled_action = sampled_action = sampled_eps * tf.exp(act_logstd) + act_mean

        # compute KL, log_prob, entropy
        N = tf.shape(obs)[0]
        p_n = normal_log_prob(action, act_mean, act_logstd, act_dim)
        oldp_n = normal_log_prob(action, oldact_mean, oldact_logstd, act_dim)
        ratio_n = tf.exp(p_n - oldp_n)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()
        kl = tf.reduce_mean(normal_kl(oldact_mean, oldact_logstd, act_mean, act_logstd))
        ent = tf.reduce_mean(normal_entropy(act_logstd))

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)

        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_mean(normal_kl(tf.stop_gradient(act_mean),
                                                 tf.stop_gradient(act_logstd),
                                                 act_mean, act_logstd))
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())

    def act(self, obs, *args):
        if len(obs.shape) > 1:
            obs = np.squeeze(obs)
        obs = np.expand_dims(obs, 0)
        obs_new = obs #np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        self.prev_obs = obs

        action, mean, logstd = self.session.run([self.sampled_action, self.act_mean, self.act_logstd], {self.obs: obs_new})
        self.prev_action = action
        action = action[0]
        return action, mean, logstd, np.squeeze(obs_new)

    def learn(self, max_iters = 1000, animate = True):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        for i in range(max_iters):
            print("\n********** Iteration %i ************" % i)
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch,
                animate=(animate and i % 10 == 0))

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_mean_n = np.concatenate([path["action_means"] for path in paths])
            action_logstd_n = np.concatenate([path["action_logstds"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()
            advant_n /= (advant_n.std() + tiny)
            
            feed = {self.obs: obs_n,
                    self.action: action_n,
                    self.advant: advant_n,
                    self.oldact_mean: action_mean_n,
                    self.oldact_logstd: action_logstd_n}


            episoderewards = np.array([path["rewards"].sum() for path in paths])

            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                # Computing baseline function for next iter.
                self.vf.fit(paths)
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                items = ["Total number of episodes", "Average sum of rewards per episode",
                         "Entropy","Baseline explained", "KL between old and new distribution",
                         "Surrogate loss", "Time elapsed"]
                stats[items[0]] = numeptotal
                stats[items[1]] = episoderewards.mean()
                stats[items[2]] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats[items[3]] = exp
                stats[items[4]] = kloldnew
                stats[items[5]] = surrafter
                stats[items[6]] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                for k in items:
                    v = stats[k]
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                #if exp > 0.8:
                #    self.train = False

tf.reset_default_graph()
training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    #task = "RepeatCopy-v0"
    task = "Pendulum-v0"

hdlr = logging.FileHandler('./log/'+task)
logging.getLogger().addHandler(hdlr)

env = envs.make(task)
#env.monitor.start(training_dir)

#env = SpaceConversionEnv(env, Box, Discrete)

agent = TRPOAgent(env)
agent.learn(10000, False)
#env.monitor.close()
#gym.upload(training_dir,
#           algorithm_id='trpo_ff')


