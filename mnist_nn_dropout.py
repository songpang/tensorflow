# mnist_nn_deep.py
# xavier initializer

import tensorflow as tf
old_v = tf.logging.get_verbosity()          # warning
tf.logging.set_verbosity(tf.logging.ERROR)  # warning (Please use alternatives such as official/mnist/dataset.py from tensorflow/models.)
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('Data/mnist', one_hot=True)

print(type(mnist))
print(type(mnist.train))

learning_rate = 0.001
training_epcohs = 15
batch_size = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

keep_probe = tf.placeholder(tf.float32)

# W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
W1 = tf.get_variable("W1", shape=[784,512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_probe)

# W2 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob = keep_probe)

# W3 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias1')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob = keep_probe)

# W4 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name='bias2')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
layer4 = tf.nn.dropout(layer1, keep_prob = keep_probe)

# W5 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name='bias1')

logits = tf.matmul(layer4, W5) + b5 # (?, 784) * (784, 10) = (?, 10)
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for epoch in range(training_epcohs):
    avg_cost = 0
    # 550 = 55000/100(batch_size)
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_probe:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch: ', '%04d'%(epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

print("Learning Finished!!")

# Test model and check accuracy
# accuracy computation

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                  dtype = tf.float32))

a = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_probe:1})

print("\nAccuracy", a)

# get one random test data and predict

r = random.randint(0, mnist.test.num_examples - 1)
print("random = ", r, "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))

print("Prediction :", sess.run(tf.arg_max(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1], keep_probe:1}))

# matplotlib : imshow()

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
           cmap="Greys", interpolation='nearest') # 2차원 보간법

plt.show()