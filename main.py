import numpy as np
from keras.engine.saving import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Saves the model
def save_model (network: Sequential):
    # serialize model to JSON
    model_json = network.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    network.save_weights("network_weights.h5")

    print("Saved model to disk")


# Loads the model
def load_model ( ):
    # load JSON and create model
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("network_weights.h5")
    print("Loaded model from disk")
    return loaded_model


# Prepares the data
def data_preprocessing ( ):
    # Download the MNIST dataset
    dataset = datasets.fetch_openml("mnist_784")

    # Reshape the data to a (70000, 28, 28) tensor
    data = dataset.data.reshape((dataset.data.shape[0]), 28, 28)

    # Reshape the data to a (70000, 28, 28, 1) tensor
    data = data[:, :, :, np.newaxis]

    # Scale values from range of [0-255] to [0-1]
    scaled_data = data / 255.0

    # Split the dataset into training and test sets
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        scaled_data,
        dataset.target.astype("int"),
        test_size = 0.33)

    # Transform training labels to one-hot encoding
    trainLabels = np_utils.to_categorical(trainLabels)

    # Transform test labels to one-hot encoding
    testLabels = np_utils.to_categorical(testLabels)

    return trainData, testData, trainLabels, testLabels


# Trains the model, please use solely to Reset
def train_model (trainData, testData, trainLabels, testLabels, epochs):
    model = Sequential()
    # Add the first convolution layer
    model.add(Convolution2D(
        filters = 20,
        kernel_size = (5, 5),
        padding = "same",
        input_shape = (28, 28, 1)))

    # Add a ReLU activation function
    model.add(Activation(
        activation = "relu"))

    # Add a pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))

    # Add the second convolution layer
    model.add(Convolution2D(
        filters = 50,
        kernel_size = (5, 5),
        padding = "same"))

    # Add a ReLU activation function
    model.add(Activation(
        activation = "relu"))

    # Add a second pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))

    # Flatten the network
    model.add(Flatten())

    # Add a fully connected hidden layer
    model.add(Dense(500))

    # Add a ReLU activation function
    model.add(Activation(
        activation = "relu"))

    # Add a fully-connected output layer
    model.add(Dense(10))

    # Add a softmax activation function
    model.add(Activation("softmax"))

    # Compile the network
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = SGD(lr = 0.01),
        metrics = ["accuracy"])

    # Train the model
    model.fit(
        trainData,
        trainLabels,
        batch_size = 128,
        epochs = epochs,
        verbose = 1)

    # Evaluate the model
    (loss, accuracy) = model.evaluate(
        testData,
        testLabels,
        batch_size = 128,
        verbose = 1)

    save_model(model)
    # Print the model's accuracy
    print(accuracy)


# Returns the predicted digit
def predict_digit (imagesArray, batch_size = None, verbose = 0):
    model: Sequential = load_model()
    return model.predict_classes(x = imagesArray, batch_size = batch_size, verbose = verbose)


# Returns the probability of each image showing a certain Digit
def predict_digit_probability (imagesArray, batch_size = 32, verbose = 0):
    model: Sequential = load_model()
    return model.predict_proba(x = imagesArray, batch_size = batch_size, verbose = verbose)


if __name__ == "__main__":
    train_model(*data_preprocessing(), epochs = 30)
    # train_data, test_data, train_labels, test_labels = data_preprocessing()
    # train_model(train_data, test_data, train_labels, test_labels, 30) is a valid alternative
