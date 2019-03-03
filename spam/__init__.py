# MNIST

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholder for the input image data
# The placeholder is a tensor, so we set the data type(float32), and its shape
# the use of none means that the existence of an dimension is know but not its magnitude
# the second dimension has capacity to keep 784 itens(each image is 28 by 28, which gives
# 784 pixels)
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

# y bar : is an 10 element vector that holds the predicted probability of each digit(0-9)
# class
y_ = tf.placeholder(tf.float32, [None, 10])

# Definition of weights and balances
W = tf.Variable(tf.zeros([784, 10]))  # 784 pixels, 10 digits
b = tf.Variable(tf.zeros([10]))  # 10 digits

# Defining the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss is measured by Cross Entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
)

# Gradient descent used to minimize the lost
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5)\
    .minimize(cross_entropy)

# Initialize the global variables
init = tf.global_variables_initializer()

# Define a Session
sess = tf.Session()

# Set up the initialization of all global variables
sess.run(init)

# Perform 1000 training steps
for i in range(1000):
    # batch_xs=image, batch_ys=digit(0-9)
    batch_xs, batch_ys = mnist.train.next_batch(100)  # get 100 random data points

    # Do the optimization with this data
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating how well the model did, comparing the highest probability in
# actual (y) and predicted(y_)
correct_prediction = tf.equal(x=tf.argmax(y, 1), y=tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test accuracy: {0}%".format(test_accuracy * 100.0))


# Closing the session
sess.close()
