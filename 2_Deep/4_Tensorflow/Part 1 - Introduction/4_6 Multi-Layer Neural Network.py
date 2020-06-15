import numpy as np
import tensorflow as tf

# Set the random seed so things are reproducible
tf.random.set_seed(7)

# Create 3 random input features
features = tf.random.normal((1, 3))

# Define the size of each layer in our network
# Number of input units, must match number of input features
n_input = features.shape[1]
n_hidden = 2                    # Number of hidden units
n_output = 1                    # Number of output units

# Create random weights connecting the inputs to the hidden layer
W1 = tf.random.normal((n_input, n_hidden))

# Create random weights connecting the hidden layer to the output layer
W2 = tf.random.normal((n_hidden, n_output))

# Create random bias terms for the hidden and output layers
B1 = tf.random.normal((1, n_hidden))
B2 = tf.random.normal((1, n_output))


def sigmoid_activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: tf.Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    """
    return 1/(1+tf.exp(-x))


# For one layer:
#y = sigmoid_activation(tf.matmul(features,weights,transpose_b = True) + bias)

# y = f2(f1(xW1)W2)

#Udacity calls output_hidden simply "h"
output_hidden = sigmoid_activation(
    tf.matmul(features, W1) + B1)
output = sigmoid_activation(
    tf.matmul(output_hidden, W2) + B2)

print(output)
