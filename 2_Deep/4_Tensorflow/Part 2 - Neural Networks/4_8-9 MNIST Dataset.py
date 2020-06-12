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
print('There are {:,} classes in our dataset'.format(num_classes))

num_training_examples = dataset_info.splits['train'].num_examples
print('\nThere are {:,} images in the training set'.format(
    num_training_examples))

for image, label in training_set.take(1):
    print('The image #1 in the training set has:')
    print('\u2022 dtype:', image.dtype)
    print('\u2022 shape:', image.shape)

    print('\nThe labels of the images have:')
    print('\u2022 dtype:', label.dtype)

    image = image.numpy().squeeze()  # squeeze reduces 3D [28,28,1] to [28,28]
    label = label.numpy()

    # Plot the image
plt.imshow(image, cmap=plt.cm.binary)  # binary = grayscale
plt.colorbar()
plt.show()

print('The label of this image is:', label)

# CREATE PIPELINE
def normalize(image, label):    #map pixel data to the scale [0,1]
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

# Floor Division(//): Divides and returns an integer, dumping the digits after the decimal.
training_batches = training_set.cache().shuffle(
    num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)


for image_batch, label_batch in training_batches.take(1):   #one batch (64 images)
    print('The images in each batch have:')
    print('\u2022 dtype:', image_batch.dtype) 
    print('\u2022 shape:', image_batch.shape)
  
    print('\nThere are a total of {} image labels in this batch:'.format(label_batch.numpy().size))
    print(label_batch.numpy())
    images = image_batch.numpy().squeeze()
    labels = label_batch.numpy()

# Plot the image
plt.imshow(images[0], cmap = plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is:', labels[0])