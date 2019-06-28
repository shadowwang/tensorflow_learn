import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Hello = tf.constant('Hello TensorFlow!')
sess = tf.Session()
print(sess.run(Hello))
