import tensorflow as tf
import math


def generate_batch_memory_task(sz_dataset, sz_batch, to_remember_len, blank_separation_len):
    """
    Generate a batch for the memory task.
    Format:
    Input: m^to_remember_len blank^blank_separation_len # blank^to_remember_len
    Label: blank^(to_remember_len + blank_separation_len + 1) m^to_remember_len

    First to_remember_len times a random state chosen from 8 possible states
    Then blank_separation_len blank states
    Then delimiter state
    Finally, blank space for memory

    Memory task: Output blanks followed by the memory states

    :param sz_dataset: Size of the dataset
    :param sz_batch: Batch size
    :param to_remember_len: Nbr of states to remember
    :param blank_separation_len: Nbr of blanks used to test the memory length
    :return: Dataset for the memory task
    """
    blank = 8
    separator = 9

    # Random memory
    data = tf.data.Dataset.from_tensor_slices(tf.random_uniform([sz_dataset, to_remember_len], maxval=8,
                                                                dtype=tf.int32))
    # Add blanks and delimiter
    data = data.map(lambda x: tf.concat([x, blank * tf.ones([blank_separation_len], dtype=tf.int32),
                                         separator * tf.ones([1], dtype=tf.int32)], axis=0))
    # Add blanks
    data = data.map(lambda x: tf.concat([x, blank * tf.ones([to_remember_len], dtype=tf.int32)], axis=0))
    # One hot encoding and add target series
    data = data.map(lambda x: (tf.one_hot(x, 10), tf.concat(
      [blank * tf.ones([to_remember_len + blank_separation_len + 1], dtype=tf.int32), x[:to_remember_len]], axis=0)))
    data = data.batch(sz_batch)
    return data


def generate_sin_data(sz_dataset, sz_batch, num_samples=100, sample_step=0.15, return_start=False):
    """ Generate dataset for the sin learning task.
    Dataset contains a sin function evaluated at multiple equidistant evaluating points given a random start point.
    Labels correspond to the value of the following evaluation point.

    :param sz_dataset: Size of the dataset
    :param sz_batch: Size of the batch
    :param num_samples: Number of sampling points
    :param sample_step: Distance between sampling points
    :param return_start: If True, add starting point and frequency
    :return:
    """
    # Random start points and frequencies
    min_freq = 1.0
    max_freq = 2.0
    data = tf.data.Dataset.from_tensor_slices(
        (tf.random_uniform([sz_dataset], maxval=4 * math.pi, dtype=tf.float32),  # starting point
         tf.random_uniform([sz_dataset], minval=min_freq, maxval=max_freq, dtype=tf.float32)))  # frequency

    # Add sampling points
    data = data.map(lambda s, f: (tf.lin_space(start=s,
                                               stop=s + sample_step * (num_samples + 1),
                                               num=(num_samples + 1)), s, f))
    # Apply sin to sampling points
    data = data.map(lambda x, s, f: (tf.sin(tf.divide(x, f)), s, f))

    # Add target series by shifting initial series by one
    data = data.map(lambda x, s, f: (x[:-1], x[1:]) + ((s, f) if return_start else ()))
    data = data.batch(sz_batch)

    return data
