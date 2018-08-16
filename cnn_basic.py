# cnn_basic.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype = np.float32)
print(image.shape) # (1, 3, 3, 1)
plt.imshow(image.reshape(3,3), cmap = 'Greys')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# conv2d
# image : (1, 3, 3, 1), Filter : (2, 2, 2, 1), stride : (1, 1)
# number of images : 1, 3 * 3 image, color : 1
# padding : VALID
print("image:\n", image)
print(image.shape) # (1, 3, 3, 1)

weight = tf.constant([[[[1.]], [[1.]]],
                     [[[1.]], [[1.]]]])

print("weight.shape:", weight.shape)
# weight.shape : (2, 2, 1, 1)
# 2 * 2 image, color : 1, filters : 1

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID')
conv2d_img = conv2d.eval()
print("conv2d_img.shape :" ,conv2d_img.shape) # (1, 2, 2, 1)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_image in enumerate(conv2d_img):
    print(one_image.reshape(2, 2))
    plt.subplot(1, 2, i+1)
    plt.imshow(one_image.reshape(2, 2), cmap='Greys')
plt.show()


# conv2d
# image : (1, 3, 3, 1), Filter : (2, 2, 2, 1), stride : (1, 1)
# number of images : 1, 3 * 3 image, color : 1
# padding : VALID
print("image:\n", image)
print(image.shape) # (1, 3, 3, 1)

weight = tf.constant([[[[1.]], [[1.]]],
                     [[[1.]], [[1.]]]])

print("weight.shape:", weight.shape)
# weight.shape : (2, 2, 1, 1)
# 2 * 2 image, color : 1, filters : 1

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape :" ,conv2d_img.shape) # (1, 2, 2, 1)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_image in enumerate(conv2d_img):
    print(one_image.reshape(3, 3))
    plt.subplot(1, 2, i+1)
    plt.imshow(one_image.reshape(3, 3), cmap='Greys')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# conv2d
# image : (1, 3, 3, 1) ==> output : (1, 3, 3, 3)
print("image:\n", image)
print(image.shape) # (1, 3, 3, 1)

weight = tf.constant([[[[1.]], [[1.]]],
                     [[[1.]], [[1.]]]])

print("weight.shape:", weight.shape)
# weight.shape : (2, 2, 1, 1)
# 2 * 2 image, color : 1, filters : 1

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape :" ,conv2d_img.shape) # (1, 2, 2, 1)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_image in enumerate(conv2d_img):
    print(one_image.reshape(3, 3))
    plt.subplot(1, 2, i+1)
    plt.imshow(one_image.reshape(3, 3), cmap='Greys')
plt.show()


