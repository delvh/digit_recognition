from time import time

import tflearn
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression


# defines path to TensorBoard log files
log_path = "./tb_logs/"

start_time = time()

image_rows = 28
image_cols = 28

mnist = read_data_sets("MNIST_data/", one_hot = True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows, image_cols, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

num_classes = 10
keep_prob = 0.5

# define the shape of the data coming into the neural network
inputData = input_data(shape = [None, 28, 28, 1], name = "input")

# convolution layer 1
# do convolution on images, add bias and use RELU activation function
network = conv_2d(inputData, nb_filter = 32, filter_size = 3, activation = "relu", regularizer = "L2")
# name does not need to be set as tflearn takes care of that
# take results and run them through max_pool
network = max_pool_2d(network, 2)

# convolution layer 2
# do convolution on images, add bias and use RELU activation function
network = conv_2d(inputData, nb_filter = 64, filter_size = 3, activation = "relu", regularizer = "L2")
# take results and run them through max_pool
network = max_pool_2d(network, 2)

# fully connected layer
network = fully_connected(network, 128, activation = "tanh")

# dropout some neurons to avoid overfitting
network = dropout(network, keep_prob)

# readout layer
network = fully_connected(network, 10, activation = "softmax")

# set loss and optimizer
network = regression(network, optimizer = "adam", learning_rate = 0.01, loss = "categorical_crossentropy", name = "target")

# training
num_epoch = 2
model = tflearn.DNN(network, tensorboard_verbose = 3)
model.fit({"input": train_images}, {"target": mnist.train.labels}, n_epoch = num_epoch,
          validation_set = ({"input": test_images}, {"target": mnist.test.labels}),
          show_metric = True, run_id = "MNIST TFLearn")
print("duration: {0:.2f}s".format(time() - start_time))
