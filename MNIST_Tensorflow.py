from time import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# defines path to TensorBoard log files
log_path = "./tb_logs/"

start_time = time()
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# adds summary statistics for use in TensorBoard
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("standard deviation", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        # shows the distribution throughout time
        tf.summary.histogram("histogram", var)


# using interactive session makes it the default session so we do not need to pass sess
sess = tf.InteractiveSession()

# define placeholders for mnist input data
with tf.name_scope("MNIST Input"):
    x = tf.placeholder(tf.float32, shape = [None, 784], name = "x")
    y_ = tf.placeholder(tf.float32, shape = [None, 10], name = "y_")

# change the MNIST input data from a list of values to a 28 X 28 pixel greyscale value cube
with tf.name_scope("Input reshape"):
    x_image = tf.reshape(x, [-1, 28, 28, 1], name = "x_image")
    # allows TensorBoard to display 6 sample images in the "images" tab
    tf.summary.image("input image", x_image, 6)


# define helper functions to create weight- and bias-variables, convolution functions and pooling layers - RELU as activation function
def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev = 0.1, name = name)
    return tf.Variable(initial)


def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape = shape, name = name)
    return tf.Variable(initial)


# Convolution and pooling
def conv2d(tensor, W, name = None):
    return tf.nn.conv2d(tensor, W, strides = [1, 1, 1, 1], padding = "SAME", name = name)


def max_pool_2x2(tensor, name = None):
    return tf.nn.max_pool(tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name = name)


# define layers of the NN

# 1st convolution layer
with tf.name_scope("Convolution 1"):
    # 32 features for each 5x5 patch of the image
    with tf.name_scope("weights"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1)
    with tf.name_scope("biases"):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
    # apply convolution on images, add bias and use RELU activation
    conv1_wx_b = conv2d(x_image, W_conv1, name = "conv2d")
    tf.summary.histogram("conv1_wx_b", conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name = "relu")
    tf.summary.histogram("h_conv1", h_conv1)
    # take results and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolution layer
with tf.name_scope("Convolution 2"):
    # process the 32 features from Conv. layer 1, in 5x5 patch. Return 64 feature weights and biases
    with tf.name_scope("weights"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2)
    with tf.name_scope("biases"):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
    # do convolution of the output of the first convolution layer. Pool results.
    conv2_wx_b = conv2d(h_pool1, W_conv2, "conv2d") + b_conv2
    tf.summary.histogram("conv2_wx_b", conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b)
    tf.summary.histogram("h_conv2", h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
with tf.name_scope("fully connected"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # Connect output of pooling layer 2 as input to full connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)  # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)

# readout layer
with tf.name_scope("readout"):
    W_fc2 = weight_variable([1024, 10], name = "weight")
    b_fc2 = bias_variable([10], name = "bias")

# define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("cross entropy"):
    # loss measurement
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))

with tf.name_scope("loss optimizer"):
    # loss optimization
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # what is correct
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    # how accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross entropy sc2", cross_entropy)
tf.summary.scalar("training accuracy", accuracy)

# TensorBoard - merge summaries
summarize_all = tf.summary.merge_all()

# initialise all variables
sess.run(tf.compat.v1.global_variables_initializer())

# TensorBoard - write the default graph out so that its structure can be viewed
tbWriter = tf.summary.FileWriter(log_path, sess.graph)
# train the model

# define the number of steps and how often we display progress
num_steps = 3000
display_every = 100

start_time_model = time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    # periodic status display
    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time()
        print("step{0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - start_time_model, train_accuracy * 100))
        # write summary to log
        tbWriter.add_summary(summary, i)

# display summary
end_time = time()
print("total training time for {0} batches: {1:.2f} seconds".format(num_steps, end_time - start_time_model))

# accuracy on test data
print("test accuracy {0:.3f}%".format(accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) * 100)

sess.close()

print("total runtime: {0}s".format(end_time - start_time))
