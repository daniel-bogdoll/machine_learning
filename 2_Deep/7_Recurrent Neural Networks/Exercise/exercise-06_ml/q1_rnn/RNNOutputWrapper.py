import tensorflow as tf


class RNNOutputWrapper:
    """ Linear output mapping from cell to output which can be applied to cell outputs

    """

    def __init__(self, sz_rnn, sz_out):
        """

        :param sz_rnn: Size of the RNN output
        :param sz_out: Size of the final output
        """

        with tf.variable_scope('RNN_Output_Wrapper'):
            self.W = tf.get_variable('W_out', [sz_rnn, sz_out], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())

            self.b = tf.get_variable('b_out', [sz_out], dtype=tf.float32,
                                     initializer=tf.zeros_initializer)

    def __call__(self, rnn_out):
        """ Apply linear mapping to each output

        :param rnn_out: List of output tensors or single tensor
        :return:
        """
        if type(rnn_out) is list:
            return [tf.matmul(o, self.W) + self.b for o in rnn_out]
        else:
            return tf.matmul(rnn_out, self.W) + self.b
