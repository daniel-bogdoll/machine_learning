#region IMPORT
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
#endregion

#region DATASET
dataset, dataset_info = tfds.load('fashion_mnist', split='train+test', as_supervised=True, with_info=True)

total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples

train_size = int(0.6 * total_examples)
val_test_size = int(0.2 * total_examples)

#training_set, validation_set, test_set = dataset

training_set = dataset.take(train_size)
test_set = dataset.skip(train_size)
validation_set = test_set.skip(val_test_size)
test_set = test_set.take(val_test_size)

num_training_examples = train_size
num_validation_examples = val_test_size
num_test_examples = num_validation_examples

print('There are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))
print('There are {:,} images in the test set'.format(num_test_examples))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
#endregion

#region PIPELINE
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batches = validation_set.cache().batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)
#endregion

#region MODEL
layer_neurons = [512, 256, 128]
dropout_rate = 0.5

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Stop training when there is no improvement in the validation loss for 10 consecutive epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Save the Model with the lowest validation loss
save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5',
                                               monitor='val_loss',
                                               save_best_only=True)

EPOCHS = 100
history = model.fit(training_batches,
                    epochs = 100,
                    validation_data=validation_batches,
                    callbacks=[early_stopping, save_best])

#Save results
t = time.time()

#HDF5 format
saved_keras_model_filepath = './{}.h5'.format(int(t))
model.save(saved_keras_model_filepath)

#TensorFlow SavedModels format
savedModel_directory = './{}'.format(int(t))
tf.saved_model.save(model, savedModel_directory)

#Reload model
reloaded_SavedModel = tf.saved_model.load(savedModel_directory)
reloaded_keras_model_from_SavedModel = tf.keras.models.load_model(savedModel_directory)

for image_batch, label_batch in testing_batches.take(1):
    prediction_1 = model.predict(image_batch)
    prediction_2 = reloaded_SavedModel(image_batch, training=False).numpy()
    difference = np.abs(prediction_1 - prediction_2)
    print(difference.max())

reloaded_keras_model_from_SavedModel.summary()
#endregion