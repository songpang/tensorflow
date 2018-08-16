# cnn_mnist.py
# 3 layers : conv2d -> maxpool -> softmax

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) #  (N, 28, 28, 1)
Y = tf.placeholder(tf.float32, shape=[None, 10])

# L1 : image (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32]), stddev=0.01)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')


