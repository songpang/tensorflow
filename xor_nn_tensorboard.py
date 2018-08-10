# xor_nn_tensorboard.py

import tensorflow as tf

x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]

y_data = [[0],
          [1],
          [1],
          [0]]


X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

w1_hist = tf.summary.histogram("weights1", W1)  # for tensorboard
b1_hist = tf.summary.histogram("biases1", b1) # for tensorboard
layer1_hist = tf.summary.histogram("layer1", layer1) # for tensorboard

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

w2_hist = tf.summary.histogram("weights2", W2)  # for tensorboard
b2 = tf.summary.histogram("biases2", b2) # for tensorboard
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis) # for tensorboard

cost = -tf.reduce_mean(Y*tf.log(hypothesis) +
                       (1-Y)*tf.log(1-hypothesis))

cost_summ = tf.summary.scalar("cost", cost)  # for tensorboard


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),
                                     dtype = tf.float32))

accuracy_summ = tf.summary.scalar("accuracy", accuracy)  # for tensorboard

summary = tf.summary.merge_all()

# create summary writer
writer = tf.summary.FileWriter('./logs/xor_logs_01')
writer.add_graph(sess.graph)

# start training
for step in range(10001):
    s, _, cost_val = sess.run([summary, train, cost], feed_dict={X:x_data, Y:y_data})
    writer.add_summary(s, global_step=step)
    if step % 20 == 0:
        print(step, cost_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)

# predict : test model

print(sess.run(predict, feed_dict = {X:x_data}))