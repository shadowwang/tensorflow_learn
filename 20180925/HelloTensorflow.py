import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = a + b
print(a.graph is tf.get_default_graph())

## 输入层到隐藏层之间的权重列表
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
## 隐藏层到输出层之间的权重列表
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

#tf.assign(w1, w2, validate_shape=False)


##初始化值
#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#sess.run(w1.initializer)
#sess.run(w2.initializer)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#print(sess.run(y))

print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1,0.4], [0.5, 0.8]]}))


sess.close()


print(init_op)


