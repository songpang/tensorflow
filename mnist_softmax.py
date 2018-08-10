# mnist_softmax.py
# simple layer

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('Data/mnist', one_hot=True)

print(type(mnist))
print(type(mnist.train))

learning_rate = 0.1

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal([784, 10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

logits = tf.matmul(X, W)