# multi_variable_linear_regression.py

import tensorflow as tf

def not_used():
    x1_data = [73., 93., 89., 96., 73.] # n. -> .은 float라는 뜻
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]

    y_data = [152., 185., 180., 196., 142.]

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)

    Y = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([1]), name='weight1')
    W2 = tf.Variable(tf.random_normal([1]), name='weight2')
    W3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = x1*W1 + x2*W2 + x3*W3 + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, W1_val, W2_val, W3_val, b_val, _ = \
            sess.run([cost, W1, W2, W3, b, train],
                     feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y: y_data})
        if step % 20 == 0:
            print(step, cost_val, W1_val, W2_val, W3_val, b_val)

    print(sess.run(hypothesis, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data}))


# x_data = [5, 3]
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

# y_data = [5, 1]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

print(sess.run(hypothesis, feed_dict={X:x_data, Y:y_data}))



