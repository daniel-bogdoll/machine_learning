import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Network(tf.keras.Model):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.num_classes = num_classes
    
        # Define layers 
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden_layer_1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.hidden_layer_2 = tf.keras.layers.Dense(64, activation = 'relu')
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation = 'softmax')
    
    # Define forward Pass   
    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
    
        return x

subclassed_model = Network(10)
subclassed_model.build((784, 28, 28, 1))
subclassed_model.summary()