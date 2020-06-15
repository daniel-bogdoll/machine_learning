#region IMPORT
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
#endregion

#region DATASET
dataset, dataset_info = tfds.load('cats_vs_dogs', split='train', as_supervised=True, with_info=True)

total_examples = dataset_info.splits['train'].num_examples

train_size = int(0.6 * total_examples)
val_test_size = int(0.2 * total_examples)

training_set = dataset.take(train_size)
test_set = dataset.skip(train_size)
validation_set = test_set.skip(val_test_size)
test_set = test_set.take(val_test_size)

num_training_examples = train_size
num_validation_examples = val_test_size
num_test_examples = num_validation_examples

print('There are {:,} images in the whole set'.format(total_examples))
print('There are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))
print('There are {:,} images in the test set'.format(num_test_examples))
#endregion

#region PIPELINE
batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label


training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)
#endregion

#region MODEL
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(2, activation = 'softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#endregion

#region TRAINING
EPOCHS = 2
history = model.fit(training_batches,
                    epochs = EPOCHS,
                    validation_data=validation_batches)
#endregion
