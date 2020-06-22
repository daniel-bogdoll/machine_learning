import tensorflow as tf
import collections  # needed for namedtuple
import RNNCell  # import abstract base class


# Collection of tensors describing the current hidden cell configuration. Here only one hidden state.
MyBasicRNNState = collections.namedtuple('MyBasicRNNState', ['hidden'])  # Create new named tuple class


class BasicRNNCell(RNNCell.RNNCell):
    """ Implementation of a very basic RNN cell.

    """

    def __init__(self, n_inp, n_out):
        """

        :param n_out: Dimension of the output
        :param n_inp: Dimension of the input
        """
        super(BasicRNNCell, self).__init__(n_inp, n_out)
        self.sz_hidden = n_out

        with tf.variable_scope('BasicRNNCell'):
            #####Start Subtask 1a#####

            # initialize forward part with with Xavier (Glorot) initialization
            # initialize recurrent part with identity matrix
            w_forward_init = tf.contrib.layers.xavier_initializer()((n_inp, n_out))
            w_recurrent_init = tf.eye(n_out, n_out)
            w_init = tf.concat([w_forward_init, w_recurrent_init], axis=0)
            self.W = tf.get_variable('W', initializer=w_init)
            self.b = tf.get_variable('b', [n_out], initializer=tf.zeros_initializer)
            #####End Subtask#####

    def __call__(self, state, x):
        """ Apply the cell to an given input i.e. unroll the RNN on the input and add it to the graph.

        :param state: Initial state
        :param x: Input, shape: (batch size, time steps, input dimension)
        :return: List of outputs of every step, tensor of the final state after computation
        """
        # In case one-hot encoding is used
        x = tf.cast(x, tf.float32)

        #####Start Subtask 1b#####

        # Split input into inputs for every time step
        x = tf.unstack(x, axis=1)
        outputs = []

        # Unroll the net on the inputs
        for x_i in x:
            # Handle 1 dimensional inputs
            if len(x_i.get_shape()) < 2:
                x_i = tf.expand_dims(x_i, axis=1)

            # Apply cell
            comb_inp = tf.concat([x_i, state.hidden], axis=1)
            state = MyBasicRNNState(tf.nn.relu(tf.matmul(comb_inp, self.W) + self.b))

            outputs.append(state.hidden)

        #####End Subtask#####

        return outputs, state

    def get_state_placeholder(self, name):
        """

        :param name: Name of the placeholder
        :return: Placeholder for the state
        """
        ph_hidden = tf.placeholder(dtype=tf.float32, shape=(None, self.sz_hidden), name= '%s_%s' % (name, 'hidden'))
        return MyBasicRNNState(ph_hidden)

    def get_zero_state_for_inp(self, x):
        """ Return a initial zero state for the cell based on the expected batch size.

        :param x: Expected input (Used to determine the batch size)
        :return: Zero initial state for the expected batch size
        """
        zeros_dims = tf.stack([tf.shape(x)[0], self.sz_hidden]) 
        return MyBasicRNNState(tf.zeros(zeros_dims))

    def create_state_feed_dict(self, pl_state, np_state):
        """ Return a dict which contains contains state assignments for tensorflows feed_dict

        :param pl_state: Data structure containing the state placeholders (only hidden state)
        :param np_state: Data for the state placeholders
        :return: Dict feeding the state data into the placeholders
        """
        return {pl_state.hidden: np_state.hidden}
