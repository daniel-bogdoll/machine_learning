import data_generator
import tensorflow as tf
import model
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_test_sin(sess, cell, output_wrapper, nbr_samples, rnn_type, sample_step=0.15):

    # insert your code here

