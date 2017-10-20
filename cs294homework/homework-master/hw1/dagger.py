import tensorflow as tf
import numpy as np
import behaviorcloning as bc
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as learn
import returnobs
import expertmark
import evaluation
import pickle


expert_data = bc.expert('experts/Humanoid-v1.pkl','Humanoid-v1',num_rollouts=100)
observations = expert_data["observations"]
actions = expert_data["actions"]
actions = np.squeeze(actions)

_, num_actions = np.shape(actions)
_, dim = np.shape(observations)

feature_columns = [layers.real_valued_column("",dimension=dim)]

classifier = learn.DNNRegressor(feature_columns=feature_columns,
                                hidden_units=[2048,256],
                                label_dimension=num_actions,
                                optimizer=tf.train.AdamOptimizer)

for i in np.arange(4):

    # classifier.fit(x=observations,y=actions,steps=2000,batch_size=50)
    classifier.fit(x=observations,y=actions,steps=2000)

    return_obs = returnobs.cloning(classifier,'Humanoid-v1',num_rollouts=20)
    return_act = expertmark.expertactions('experts/Humanoid-v1.pkl',return_obs)
    return_act = np.squeeze(return_act)

    observations = np.concatenate((observations,return_obs))
    actions = np.concatenate((actions,return_act))

expert_data = evaluation.cloning(classifier,'Humanoid-v1',num_rollouts=20)





















