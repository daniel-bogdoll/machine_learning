import tensorflow as tf
import collections
import RNNCell

MyLSTMState = collections.namedtuple('MyLSTMState', ['cell', 'hidden'])


class MyLSTMCell(RNNCell.RNNCell):
    """ Implementation of a basic LSTM cell.

    """

    def __init__(self, n_inp, n_lstm):

    # insert your code here

