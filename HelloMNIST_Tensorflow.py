from time import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

start_time = time()
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# using interactive session makes it the default session so we do not need to pass sess
sess = tf.InteractiveSession()

# define placeholders for mnist input data
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# change the MNIST input data from a list of values to a 28 X 28 pixel greyscale value cube
x_image = tf.reshape(x, [-1, 28, 28, 1], name = "x_image")  #


# define helper functions to create weight- and bias-variables, convolution functions and pooling layers - RELU as activation function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


# Convolution and pooling
def conv2d(tensor, W):
    return tf.nn.conv2d(tensor, W, strides = [1, 1, 1, 1], padding = "SAME")


def max_pool_2x2(tensor):
    return tf.nn.max_pool(tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


# define layers of the NN

# 1st convolution layer
# 32 features for each 5x5 patch of the image
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolution layer
# process the 32 features from Conv. layer 1, in 5x5 patch. Return 64 feature weights and biases
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# do convolution of the output of the first convolution layer. Pool results.
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Connect output of pooling layer 2 as input to full connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)  # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))

# loss optimization
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# what is correct
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# how accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialise all variables
sess.run(tf.compat.v1.global_variables_initializer())

# train the model

# define the number of steps and how often we display progress
num_steps = 3000
display_every = 100

start_time_model = time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict = { x: batch[0], y_: batch[1], keep_prob: 0.5 })

    # periodic status display
    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict = { x: batch[0], y_: batch[1], keep_prob: 1.0 })
        end_time = time()
        print("step{0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - start_time_model, train_accuracy * 100))

# display summary
end_time = time()
print("total training time for {0} batches: {1:.2f} seconds".format(num_steps, end_time - start_time_model))

# accuracy on test data
print("test accuracy {0:.3f}%".format(accuracy.eval(feed_dict = { x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0 })) * 100)

sess.close()

print("total runtime: {0}s".format(end_time - start_time))
