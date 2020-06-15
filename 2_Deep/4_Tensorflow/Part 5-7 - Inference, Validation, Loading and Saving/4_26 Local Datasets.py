# region IMPORT
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available()
      else '\t\u2022 GPU device not found. Running on CPU')
# endregion

# region DATASET
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file(
    'cats_and_dogs_filterted.zip', origin=_URL, extract=True)
# endregion

# region PIPELINE
BATCH_SIZE = 64
IMG_SHAPE = 224

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# image_gen = ImageDataGenerator(rescale=1./255)  #rescale pixel values from 0 to 1
#
# one_image = image_gen.flow_from_directory(directory=train_dir,
#                                          batch_size=BATCH_SIZE,
#                                          shuffle=True,
#                                          target_size=(IMG_SHAPE,IMG_SHAPE),
#                                          class_mode='binary')
#
# plt.imshow(one_image[0][0][0])
# plt.show()
# endregion


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# region DATA AUGMENTATION
#image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
#image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
#image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
# Combinations
#image_gen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,rotation_range=45,zoom_range=0.5)

image_gen = ImageDataGenerator(rescale=1./255,rotation_range=45, width_shift_range=0.2, height_shift_range=0.2,
                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

train_data_gen = image_gen.flow_from_directory(directory=train_dir,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               target_size=(
                                                   IMG_SHAPE, IMG_SHAPE),
                                               class_mode='binary')

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(directory=validation_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')
                                                 
#augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#plotImages(augmented_images)
#endregion

#region MODEL
layer_neurons = [1024, 512, 256, 128, 56, 28, 14]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (IMG_SHAPE, IMG_SHAPE, 3)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
            
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
#endregion

#region Training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10

history = model.fit_generator(train_data_gen,
                              epochs=EPOCHS,
                              validation_data=val_data_gen)
#endregion

#region CNN Convolutional Neural Network
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/


#endregion
