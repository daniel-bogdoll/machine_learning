import tensorflow as tf
import collections
import RNNCell

MyLSTMState = collections.namedtuple('MyLSTMState', ['cell', 'hidden'])


class MyLSTMCell(RNNCell.RNNCell):
    """ Implementation of a basic LSTM cell.

    """

    def __init__(self, n_inp, n_lstm):
        """

        :param n_inp:  Dimension of the input
        :param n_lstm: Size of lstm cell
        """
        super(MyLSTMCell, self).__init__(n_inp, n_lstm)
        self.n_lstm = n_lstm

        with tf.variable_scope('LSTM_Cell'):
            #####Start Subtask 1c#####

            # Forget gate
            self.W_f = tf.get_variable('W_f', [n_inp + n_lstm, n_lstm], dtype=tf.float32, initializer=None)
            self.b_f = tf.get_variable('b_f', [n_lstm], dtype=tf.float32, initializer=tf.zeros_initializer)

            # Input gate
            self.W_i = tf.get_variable('W_i', [n_inp + n_lstm, n_lstm], dtype=tf.float32, initializer=None)
            self.b_i = tf.get_variable('b_i', [n_lstm], dtype=tf.float32, initializer=tf.zeros_initializer)

            # Candidate values
            self.W_c = tf.get_variable('W_c', [n_inp + n_lstm, n_lstm], dtype=tf.float32, initializer=None)
            self.b_c = tf.get_variable('b_c', [n_lstm], dtype=tf.float32, initializer=tf.zeros_initializer)

            # Output gate
            self.W_o = tf.get_variable('W_o', [n_inp + n_lstm, n_lstm], dtype=tf.float32, initializer=None)
            self.b_o = tf.get_variable('b_o', [n_lstm], dtype=tf.float32, initializer=tf.zeros_initializer)

            #####End Subtask#####

    def __call__(self, state, x):
        """ Apply the cell to an given input i.e. unroll the RNN on the input and add it to the graph.

        :param state: Initial state
        :param x: Input (Batch Size, Time steps, Input dimension)
        :return: List of outputs of every step, tensor of the final state after computation
        """
        # In case one-hot encoding is used
        x = tf.cast(x, tf.float32)

        #####Start Subtask 1d#####

        x = tf.unstack(x, axis=1)
        outputs = []

        for x_t in x:
            if len(x_t.get_shape()) < 2:
                x_t = tf.expand_dims(x_t, axis=1)

            # Concatenate hidden state with input x
            comb_inp = tf.concat([state.hidden, x_t], axis=1)
            # Forget gate - decide which parts of old cell state to keep
            f = tf.sigmoid(tf.matmul(comb_inp, self.W_f) + self.b_f)
            # Input gate - decide which values to update
            i = tf.sigmoid(tf.matmul(comb_inp, self.W_i) + self.b_i)
            # Candidate values - could be added to the cell state
            candidate = tf.tanh(tf.matmul(comb_inp, self.W_c) + self.b_c)
            # Update cell state
            cell_state = f * state.cell + i * candidate
            # Cell output
            o = tf.sigmoid(tf.matmul(comb_inp, self.W_o) + self.b_o)
            # Update hidden state
            hidden = o * tf.tanh(cell_state)

            state = MyLSTMState(cell=cell_state, hidden=hidden)
            outputs.append(hidden)

        #####End Subtask#####
        return outputs, state

    def get_state_placeholder(self, name):
        """

        :param name: Name of the placeholder
        :return: Placeholder for the state (cell and hidden state)
        """
        ph_cell = tf.placeholder(dtype=tf.float32, shape=(None, self.n_lstm), name='%s_%s' % (name, 'cell'))
        ph_hidden = tf.placeholder(dtype=tf.float32, shape=(None, self.n_lstm), name='%s_%s' % (name, 'hidden'))
        return MyLSTMState(ph_cell, ph_hidden)

    def get_zero_state_for_inp(self, x):
        """ Return a initial zero state for the cell based on the expected batch size.

        :param x: Expected input (Used to determine the batch size)
        :return: Zero initial state for the expected batch size
        """
        zeros_dim = tf.stack([tf.shape(x)[0], self.n_lstm])
        return MyLSTMState(tf.zeros(zeros_dim), tf.zeros(zeros_dim))

    def create_state_feed_dict(self, pl_state, np_state):
        """ Return a dict which contains contains state assignments for tensorflows feed_dict

        :param pl_state: Data structure containing the state placeholders (cell and hidden state)
        :param np_state: Data for the state placeholders
        :return: Dict feeding the state data into the placeholders
        """
        return {pl_state.cell: np_state.cell, pl_state.hidden: np_state.hidden}
