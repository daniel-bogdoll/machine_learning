# Usage:   python predict.py /path/to/image saved_model
# Options: python predict.py /path/to/image saved_model --top_k
#         python predict.py /path/to/image saved_model --category_names map.json

#region IMPORT
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import json
import time

from PIL import Image
#endregion

#region ARGPARSE
parser = argparse.ArgumentParser(
    description='Deep Learning',
)

#python predict.py ./test_images/cautleya_spicata.jpg my_model.h5 --top_k 5 --category_names label_map.json 

parser.add_argument('image', action="store")
parser.add_argument('model', action="store")
parser.add_argument('--top_k', action="store", dest="k", type=int)
parser.add_argument('--category_names', action="store", dest="category_names")

results = parser.parse_args()
image_path = results.image
model_path = results.model
k = results.k
category_names_file = results.category_names
#endregion

#region DATASET
dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)
training_set = dataset['train']
test_set = dataset['test']
validation_set = dataset['validation']
#endregion

#region MODEL
model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
#endregion

#region UTILITIES
with open(category_names_file, 'r') as f:
    class_names = json.load(f)

def process_image(imageArray):
    image = tf.cast(imageArray, tf.float32)
    image = tf.image.resize(imageArray, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path,model,k):
    image = Image.open(image_path)
    imageArray = np.asarray(image)
    resizedImage = process_image(imageArray)
    
    resizedImage = np.expand_dims(resizedImage, axis=0)
    probabilities = model.predict(resizedImage)
    predictions = np.zeros(shape=(probabilities.size,2))
    for i in range(probabilities.size):
        element = [int(i)+1, probabilities[0][i]]
        predictions[i] = element
    
    #print(predictions)
    sortedPredictions = sorted(predictions,key=lambda x: x[1], reverse=True)
    shortenedPredictions = sortedPredictions[:k]
    
    classes = [row[0] for row in shortenedPredictions]
    probs = [row[1] for row in shortenedPredictions]
    
    classes = np.asarray(classes).astype(int)
    
    return probs, classes
#endregion

#region PREDICTION
inputImage = Image.open(image_path)
probs, classes = predict(image_path, model, k)
classNames = [""  for x in range(k)]
for i in range(len(classes)):
    name = class_names[np.array2string(classes[i])]
    classNames[i] = name

print ("Probabilities:", probs)
print ("Classes:", classNames)
#endregion


