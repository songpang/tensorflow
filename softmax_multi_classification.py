# softmax_multi_classification.py

import tensorflow as tf

# x_data : [N, 4] --> [8, 4]
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# y_data : [N, 3] --> [8, 3]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

nb_classes = 3  # 분류갯수

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)


# 방법 1 : log함수를 이용하여 수식을 직접 표현
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))   ### axis = 1 행 0 열

# 방법 2 : softmax_cross_entropy_with_logits()함수 사용
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(2001):
    _, cost_val, W_val, b_val = sess.run([optimizer, cost, W, b],
                                         feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, "\n", W_val, b_val)


# predict
a = sess.run(hypothesis, feed_dict={X:[[1, 2, 1, 1],
                                       [2, 1, 3, 2],
                                       [3, 1, 3, 4],
                                       [4, 1, 5, 5],
                                       [1, 7, 5, 5],
                                       [1, 2, 5, 6],
                                       [1, 6, 6, 6],
                                       [1, 7, 7, 7]]})

print(a, sess.run(tf.argmax(a, 1)))   # argmax 1: 행단위  -> one-hot
print(a, sess.run(tf.argmax(a, 0)))   # argmax 0: 열단위