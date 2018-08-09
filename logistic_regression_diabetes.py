# logistic_regression_diabetes.py

import tensorflow as tf
import numpy as np

xy = np.loadtxt('Data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)

# Using 70% data
x_data = xy[0:531, 0:-1]
y_data = xy[0:531, [-1]]


X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) +
                       (1-Y) * tf.log(hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# predict
predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y),
                                  dtype=tf.float32))


for step in range(100001):
    _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step , cost_val)


# Test 30% data
x_data = xy[531:, 0:-1]
y_data = xy[531:, [-1]]

ppp , aaa = sess.run([predict, accuracy], feed_dict={X:x_data, Y:y_data})
print(ppp,aaa)