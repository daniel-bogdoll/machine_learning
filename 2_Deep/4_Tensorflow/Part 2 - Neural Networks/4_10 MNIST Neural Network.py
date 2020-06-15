import logging
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available()
      else '\t\u2022 GPU device not found. Running on CPU')

# MNIST dataset: Handwritten digits (split=train returns only training set)
training_set, dataset_info = tfds.load(
    'mnist', split='train', as_supervised=True, with_info=True)

num_classes = dataset_info.features['label'].num_classes
num_training_examples = dataset_info.splits['train'].num_examples

#PIPELINE
def normalize(image, label):    #map pixel data to the scale [0,1]
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

# Floor Division(//): Divides and returns an integer, dumping the digits after the decimal.
training_batches = training_set.cache().shuffle(
    num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)

for image_batch, label_batch in training_batches.take(1):
    images = image_batch.numpy().squeeze()
    labels = label_batch.numpy()

features = tf.reshape(images, [images.shape[0], -1])    #-1 flattens dimensions

n_input = 784
n_hidden = 256                    
n_output = 10

W1 = tf.random.normal((n_input, n_hidden))
B1 = tf.random.normal((1, n_hidden))

W2 = tf.random.normal((n_hidden, n_output))
B2 = tf.random.normal((1, n_output))

def sigmoid_activation(x):
    return 1/(1+tf.exp(-x))

def softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=1, keepdims = True)    #ensure output tensor has correct shape

h = sigmoid_activation(tf.matmul(features, W1) + B1)
probabilities = softmax(tf.matmul(h, W2) + B2)

sum_all_prob = tf.reduce_sum(probabilities, axis = 1).numpy()

for i, prob_sum in enumerate(sum_all_prob):
    print('Sum of probabilities for Image {}: {:.1f}'.format(i+1, prob_sum))