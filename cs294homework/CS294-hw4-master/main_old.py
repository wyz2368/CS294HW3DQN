import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0,dtype=tf.float32))
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

def main_cartpole(n_iter=100, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, animate=True, logdir=None):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.initialize_all_variables().run() #pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

def main_pendulum(n_iter=4000, gamma=1.0, min_timesteps_per_batch=20000, stepsize=2e-3, animate=True, logdir=None):
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)  # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac",
                             dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)  # advantage function estimate
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(0.1)))  # hidden layer
    sy_h2 = tf.nn.relu(dense(sy_h1, 8, "h2", weight_init=normc_initializer(0.1)))  # hidden layer
    sy_mean_na = dense(sy_h1, ac_dim, 'mean', weight_init=normc_initializer(0.01))
    sy_logstd_a = dense(sy_h1, ac_dim, 'logstd', weight_init=normc_initializer(0.01))
    # sy_logstd_a = tf.get_variable("logstdev", [ac_dim], initializer=tf.constant_initializer(0,dtype=tf.float32))

    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean',
                                   dtype=tf.float32)  # mean BEFORE update (just used for KL diagnostic)
    sy_oldlogstd_a = tf.placeholder(shape=[None, ac_dim], name='oldlogstd',
                                    dtype=tf.float32) # logstd BEFORE update (just used for KL diagnostic)
    sy_sampled_eps = tf.random_normal(tf.shape(sy_mean_na))
    sy_sampled_ac = (sy_sampled_eps * tf.exp(sy_logstd_a) + sy_mean_na)[0]
    # log P = -0.5 * (x - mu) ^2 / Sigma - k/2 log(2 pi) - 0.5 log(|Sigma|)
    #       = -0.5 * (x - mu) ^ 2 / (std)^2 - sum_k log(std) - 0.5 * k * log(2pi)
    sy_logprob_n = tf.reduce_sum(0.5 * tf.square((sy_ac_n - sy_mean_na) / tf.exp(sy_logstd_a)) + sy_logstd_a,
                                 axis=1) + 0.5 * tf.cast(ac_dim,dtype=tf.float32) * np.log(2 * np.pi)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # entropy = 0.5 * ln((2pi e)^k |Sigma|) = 0.5 * [ k * log(2pie) ] + 0.5 * sum_k log(std_k ^ 2)
    #         = 0.5 * k * (log2pi + 1) + sum_k log(std_k)
    sy_ent = tf.reduce_mean(tf.reduce_sum(sy_logstd_a, axis=1) + 0.5 * ac_dim * (1 + np.log(2 * np.pi)))
    # KL = 0.5 * {trace(Sigma^-1 * Sigma_o) + (mu - mu_o)^T Sigma^{-1} (mu - mu_0) - k + ln|Sigma| - ln|Sigma_o|}
    #    = 0.5 * {sum_k std_o^2/std^2 + sum_k det_k^2/std^2 - k + 2 * sum_k log std - 2 * sum_k log std_o}
    sy_det_mu = sy_mean_na - sy_oldmean_na
    sy_std = tf.exp(sy_logstd_a)
    sy_oldstd = tf.exp(sy_oldlogstd_a)
    sy_kl = 0.5 * tf.reduce_sum(
                     tf.reduce_sum(tf.square(sy_oldstd / sy_std), axis=1) +
                     tf.reduce_sum(tf.square(sy_det_mu / sy_std), axis=1) -
                     tf.cast(ac_dim, tf.float32) +
                     2.0 * tf.reduce_sum(sy_logstd_a, axis=1) -
                     2.0 * tf.reduce_sum(sy_oldlogstd_a, axis=1),
                   )
    # <<<<<<<<<<<<<<<<<<<<<<

    sy_surr = tf.reduce_mean(sy_adv_n * sy_logprob_n)  # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[],
                                 dtype=tf.float32)  # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__()  # equivalent to `with sess:`
    tf.initialize_all_variables().run()  # pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                if len(ob.shape) > 1:
                    ob = np.squeeze(ob)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break
            path = {"observation": np.array(obs), "terminated": terminated,
                    "reward": np.array(rewards), "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, old_mean, old_logstd = sess.run([update_op, sy_mean_na, sy_logstd_a],
                                   feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n,
                                              sy_stepsize: stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no: ob_no,
                                                       sy_oldmean_na: old_mean,
                                                       sy_oldlogstd_a: old_logstd})

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

if __name__ == "__main__":
    #main_cartpole(logdir=None,animate=False) # when you want to start collecting results, set the logdir
    main_pendulum(logdir='./progress', animate=True)
