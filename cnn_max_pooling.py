import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

# max pooling (1, 2, 2, 1) ---> (1, 1, 1, 1)
# padding : VALID
image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)
print(image.shape) # (1, 2, 2, 1)

pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='VALID')

print(pool.shape)
print(pool.eval())

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# max pooling (1, 2, 2, 1) ---> (1, 2, 2, 1)
# padding : VALID
image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)
print(image.shape) # (1, 2, 2, 1)

pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')

print(pool.shape)
print(pool.eval())