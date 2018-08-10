# softmax_zoo_multi_classification.py

import tensorflow as tf
import numpy as np

xy = np.loadtxt('Data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]  ## [101, 16]
y_data = xy[:, [-1]]  ## [101, 1]

print(x_data.shape, y_data.shape)

nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_ONE_HOT = tf.one_hot(Y, nb_classes)                ## [None, 1, 7]
Y_ONE_HOT = tf.reshape(Y_ONE_HOT, [-1, nb_classes])  ##   [None, 7]

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_ONE_HOT)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy
predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(Y_ONE_HOT, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


# start training
for step in range(2001):
    _, cost_val, W_val, b_val = sess.run([optimizer, cost, W, b], feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={X:x_data, Y:y_data})

print("\nHypothesis", h, "\predict", p, "\nAccuracy", a)

# predict : test model
pred = sess.run(predict, feed_dict={X:x_data})


for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} // REAL Y: {}".format(p == int(y), p, int(y)))

# zip([1, 2], [3, 4])   -->  [1, 3], [2, 4]
# flatten() : [[0], [1], [2]] --> [0, 1, 2]

