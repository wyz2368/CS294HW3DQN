import tensorflow as tf

a = tf.constant(5)

sess = tf.Session()
print sess.run(a)