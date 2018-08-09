# new_prac.py

import tensorflow as tf
import numpy as np

xy = np.loadtxt('Data/cars.csv', delimiter=',')
x = xy[0]
y = xy[1]

X = tf.placeholder(tf.float32, shape=None)
Y = tf.placeholder(tf.float32, shape=None)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
