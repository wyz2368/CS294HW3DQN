import tensorflow as tf
import numpy as np
import behaviorcloning as bc
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as learn
import evaluation

expert_data = bc.expert('experts/Ant-v1.pkl','Ant-v1',num_rollouts=400)
observations = expert_data["observations"]
actions = expert_data["actions"]
actions = np.squeeze(actions)

_, num_actions = np.shape(actions)
_, dim = np.shape(observations)

# print ("***************************************************")
# print (num_actions,dim)
# print ("***************************************************")

feature_columns = [layers.real_valued_column("",dimension=dim)]

classifier = learn.DNNRegressor(feature_columns=feature_columns,
                                hidden_units=[128,64,32],
                                label_dimension=num_actions,
                                optimizer=tf.train.AdamOptimizer)
# def get_train_inputs():
#     x = tf.constant(observations)
#     y = tf.constant(actions)
#     return x,y

# x = tf.constant(observations)
# y = tf.constant(actions)
classifier.fit(x=observations,y=actions,steps=2000,batch_size=50)

# classifier.fit(input_fn=get_train_inputs,steps=2000,batch_size=50)
expert_data = evaluation.cloning(classifier,'Ant-v1',num_rollouts=20)


# print "***************************************************"
# print "***************************************************"
# print "***************************************************"
# print "***************************************************"
# print "***************************************************"
# a = list(classifier.predict(observations[0][None,:]))
# b = 'Predictions:{}'.format(a)
# print(np.shape(a))
# print (np.shape(a[0]))
# print "***************************************************"
# print "***************************************************"
# print "***************************************************"
# print "***************************************************"
# print "***************************************************"



