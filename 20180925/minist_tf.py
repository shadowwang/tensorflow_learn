from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/wangchun/Documents/project/machinelearn/tensorflow_learn/mnist_data",
                                  one_hot=True,
                                  source_url="http://yann.lecun.com/exdb/mnist/")

print("训练数据集大小: ", mnist.train.num_examples)
print("验证数据集大小: ", mnist.validation.num_examples)
print("测试数据集大小: ", mnist.test.num_examples)
print("训练数据示例: ", mnist.train.images[0])
print("训练数据label: ", mnist.train.labels[0])


batch_size= 100
xs, ys = mnist.train.next_batch(batch_size)

print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)
