# file_input_multi_variable_linear_regression.py

import tensorflow as tf
import numpy as np

xy = np.loadtxt('Data/data-01-test-score.csv', delimiter=',')
x_data = xy[:,0:-1] ####### 원래 xy[0:25, 0:-1] --> 행 : 앞 열 : 뒤 #######
y_data = xy[:,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))


# X = tf.placeholder(tf.float32, shape=[None, 3])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W = tf.Variable(tf.random_normal([3, 1]), name= 'weight')
# b = tf.Variable(tf.random_normal([1]), name= 'bias')
#
# hypothesis = tf.matmul(X, W) + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2001):
#     sess.run(train, feed_dict={X:x_data, Y:y_data})
#     if step % 20 == 0:
#         print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
#
# print(sess.run(hypothesis, feed_dict={X:x_data, Y:y_data}))