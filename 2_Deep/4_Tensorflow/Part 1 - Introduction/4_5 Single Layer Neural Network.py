import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)

# Set the random seed so things are reproducible
tf.random.set_seed(7) 

# Create 5 random input features
features = tf.random.normal((1, 5)) #1 row, 5 columns (vector with 5 elements)

# Create random weights for our neural network
weights = tf.random.normal((1, 5))

# Create a random bias term for our neural network
bias = tf.random.normal((1, 1))

print('Features:\n', features)
print('\nWeights:\n', weights)
print('\nBias:\n', bias)

def sigmoid_activation(x):
    """ Sigmoid activation function
    
        Arguments
        ---------
        x: tf.Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    """
    return 1/(1+tf.exp(-x))

#y = f(h)
y = sigmoid_activation(tf.reduce_sum(tf.multiply(features, weights)) + bias)

#Matrix-multiplication for improved performance
#Matrix multiplications: Number of columns in the first tensor must equal to the number of rows in the second tensor
print('Features Shape:', features.shape)
print('Weights Shape:', weights.shape)  #wrong shape, needs to be transformed
print('Bias Shape:', bias.shape)
y_fast = sigmoid_activation(tf.matmul(features,weights,transpose_b = True) + bias)

print('label:\n', y)
print('label:\n', y_fast)