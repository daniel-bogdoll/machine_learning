import tensorflow as tf
import os
import glob
import re
from functools import partial
from PIL import Image

# Mapping from class labels used in the file to our labels
_MAP_CS_TO_TR_LABEL = {24: 0, 25: 1, 26: 2}


IMAGE_SIZE = [64, 64]   # Size the image is scaled to
NUM_CLASSES = 3         # Nbr of classes we want to distinguish
NUM_EX_TRAIN = 37911    # Nbr examples in training set


def get_dataset_cs(path, num_epochs, batch_size):
    """Builds and return a tensorflow dataset

    :param path: Path of the png-files
    :param num_epochs: Dataset can be used for this number of epochs
    :param batch_size: Number of examples returned for each poll
    :return: Tensorflow dataset, tensorflow dataset only containg the names of file of sufficient size
    """

    # Fetch filenames and build initial dataset of file and labels
    file_list = glob.glob(os.path.join(path, '*.png'))
    labels = _labels_from_file_names(file_list)

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(file_list), tf.convert_to_tensor(labels)))
    #####Insert your code here for subtask 1b#####
    # Filter too small images - Filter before reading the image content

    # Parse image from filename
    dataset = dataset_filtered_names.map(partial(_parse_function, im_size=IMAGE_SIZE))
    
    #####Insert your code here for subtask 1c#####
    # Basic normalization in each channel [0..1]

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset, dataset_filtered_names


def _filter_size(filename, size_threshold):
    """Checks if image size is sufficient

    :param filename: file to check
    :param size_threshold: Number of pixel threshold
    :return: True if nbr of pixel is sufficient; false else
    """

    # Lazy opening should avoid reading the entire file
    im = Image.open(filename)
    w, h = im.size

    #####Insert your code here for subtask 1b#####


def _parse_function(filename, label, im_size):
    """Parses an image and its label for tensorflow

    :param filename: Image to parse
    :param label: Label of the image
    :param im_size: Target image size
    :return: (Image, label) pair in expected format and size
    """
    label = tf.cast(label, tf.int64)

    image_string = tf.read_file(filename)

    image_decoded = tf.image.decode_png(image_string, channels=3)
    #####Insert your code here for subtask 1d#####
    return image_resized, label


def _labels_from_file_names(filenames):
    """ Generates label list from file name list

    :param filenames: List of file names
    :return: List of labels for list of files
    """

    # Match number suffix of file name
    prog = re.compile('(\d+)\.png')

    labels = []
    for name in filenames:
        res = prog.search(name)
        id = int(res.group(1))
        # Extract label -> Ignore id % 1000 = 0 since it seems
        # to correspond to groups
        base_id = id if (id < 1000) else id // 1000
        labels.append(_MAP_CS_TO_TR_LABEL[base_id])

    return labels
