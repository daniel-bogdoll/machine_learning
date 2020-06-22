import tensorflow as tf
import collections  # needed for namedtuple
import RNNCell  # import abstract base class


# Collection of tensors describing the current hidden cell configuration. Here only one hidden state.
MyBasicRNNState = collections.namedtuple('MyBasicRNNState', ['hidden'])  # Create new named tuple class


class BasicRNNCell(RNNCell.RNNCell):
    """ Implementation of a very basic RNN cell.

    """

    def __init__(self, n_inp, n_out):

    # insert your code here

