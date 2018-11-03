import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001

LEARNING_RATE = 0.8
LEARNING_DECAY = 0.99
TRAING_STEPS = 30000
BATCH_SIZE = 100



# print("训练数据集大小: ", mnist.train.num_examples)
# print("验证数据集大小: ", mnist.validation.num_examples)
# print("测试数据集大小: ", mnist.test.num_examples)
# print("训练数据示例: ", mnist.train.images[0])
# print("训练数据label: ", mnist.train.labels[0])
#
#
# batch_size= 100
# xs, ys = mnist.train.next_batch(batch_size)
#
# print("X shape: ", xs.shape)
# print("Y shape: ", ys.shape)

def inference(input_tensor, avg_class, weights1, bias1, weights2, bias2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
        return tf.matmul(layer1, weights2) + bias2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(bias1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(bias2)

def train(minist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, bias1, weights2, bias2)

    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_average_op = variable_average.apply(tf.trainable_variables())

    average_y = inference(x, variable_average, weights1, bias1, weights2, bias2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, minist.train.num_examples, LEARNING_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    correct_predication = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: minist.validation.images, y_: minist.validation.labels}

        test_feed = {x: minist.test.images, y_: minist.test.labels}

        for i in range(TRAING_STEPS):
            if (i % 1000 == 0) :
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = minist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % (TRAING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("/Users/wangchun/Documents/project/machinelearn/tensorflow_learn/mnist_data",
                                      one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



