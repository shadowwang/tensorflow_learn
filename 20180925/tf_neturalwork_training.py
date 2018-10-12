import tensorflow as tf
from numpy.random import RandomState

##自定义损失函数
batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')

##正确的输出
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

##预测的输出
y = tf.matmul(x, w)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

##生成模拟数据
rdm = RandomState()
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for x1, x2 in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 5000

    for i in range(STEPS):
        start = (i * dataset_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w))



