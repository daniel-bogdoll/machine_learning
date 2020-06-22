import abc  # abstract base class
import tensorflow as tf


class RNNCell:
    """ Class defining the basic functionality for RNNs which is used in this implementation.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_inp, n_out):
        """
        :param n_inp: Dimension of the input
        :param n_out: Dimension of the output
        """
        self.n_inp = n_inp
        self.n_out = n_out

    def get_output_size(self):
        return self.n_out

    @abc.abstractmethod
    def __call__(self, state, x):
        """ Apply the cell to an given input, i.e. unroll the RNN on the input and add it to the graph.

       :param state: Initial state
       :param x: Input tensor of shape (Batch size, Time steps, Input dimension)
       :return: List of outputs of every step, tensor of the final state after computation
       """
        return

    @abc.abstractmethod
    def get_state_placeholder(self, name):
        """

        :param name: Name of the placeholder
        :return: Placeholder for the state
        """
        return

    def get_input_placeholder(self, len_inp, name):
        """ Return a placeholder for the expected input

        :param len_inp: Len of the input in terms of time steps
        :param name: Name of the placeholder
        :return: Placeholder for the expected input
        """
        if self.get_n_inp() > 1:
            x = tf.placeholder(dtype=tf.float32, shape=(None, len_inp, self.get_n_inp()), name=name)
        else:
            x = tf.placeholder(dtype=tf.float32, shape=(None, len_inp), name=name)
        return x

    @abc.abstractmethod
    def get_zero_state_for_inp(self, x):
        """ Return a initial zero state for the cell based on the expected batch size.

        :param x: Expected input (Used to determine the batch size)
        :return: Zero initial state for the expected batch size
        """
        return

    @abc.abstractmethod
    def create_state_feed_dict(self, pl_state, np_state):
        """ Return a dict which contains state assignments for tensorflows feed_dict

        :param pl_state: Data structure containing the state placeholders
        :param np_state: Data for the state placeholders
        :return:
        """
        return

    def get_n_inp(self):
        return self.n_inp
