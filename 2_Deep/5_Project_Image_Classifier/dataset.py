import tensorflow_datasets as tfds

dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)
