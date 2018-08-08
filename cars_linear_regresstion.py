# cars_linear_regression.py
# X:speed, Y:dist

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('Data/cars.csv', unpack = True, delimiter=',', skiprows=1)
x = xy[0]
y = xy[1]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias' )

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001) # nan = not a number
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    _, W_val, b_val = sess.run([train, W, b], feed_dict={X:x, Y:y})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W), sess.run(b))


print(sess.run(hypothesis, feed_dict={X:[30, 50]}))
print(sess.run(hypothesis, feed_dict={X:[0, 25]}))

# Visualization
def prediction(x, W, b):
    return W*x + b

plt.plot(x, y, 'ro')
plt.plot((0, 25), (0, prediction(25, W_val, b_val)))
plt.plot((0, 25), (prediction(0, W_val, b_val), prediction(25,W_val, b_val)))
plt.show()
