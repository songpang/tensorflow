# Linear_Regression.py
# Using tensorflow


import tensorflow as tf
tf.set_random_seed(777) # random seed initialization
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

# 우리가 얻어야 할 것들 W, b // 모든 값들은 내부에서 dic형태로 저장된다. 그래서 name값을 넣는것이 가능함
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias' )

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# train using placeholder

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias' )

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(10001):
#     sess.run(train, feed_dict={X:[1,2,3], Y:[1,2,3]})
#     if step % 20 == 0:
#         print(step, sess.run(cost, feed_dict={X:[1,2,3], Y:[1,2,3]})
#               , sess.run(W), sess.run(b))

# start Learning(Train)
for step in range(6001):
    cost_val, W_val, B_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:[1,2,3], Y:[1,2,3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, B_val)


# predict : test model

print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[1,2,3]}))