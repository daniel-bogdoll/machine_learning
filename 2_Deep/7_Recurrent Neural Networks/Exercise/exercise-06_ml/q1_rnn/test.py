import data_generator
import tensorflow as tf
import model
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_test_sin(sess, cell, output_wrapper, nbr_samples, rnn_type, sample_step=0.15):
    """ Test the cell which should have learned to generate sin samples

    :param sess: TensorFlow session
    :param cell: RNN cell to test
    :param output_wrapper: Linear output mapping from cell to output
    :param nbr_samples: Number of samples to generate
    :param rnn_type: string representing the type of RNN used. Used as part of the filename when writing out results
    :param sample_step: Spacing between sample points
    :return:
    """

    # Number of samples to be consumed by the net to calculate the initial state for the generation process
    nbr_init_samples = 30
    # Test points (initialization values + ground truth sin)
    data_test = data_generator.generate_sin_data(5, 5, num_samples=nbr_init_samples + nbr_samples,
                                                 sample_step=sample_step, return_start=True)

    iterator = data_test.make_initializable_iterator()

    # Ground truth series, start point and frequency
    gt_t_series, gt_t_pred, gt_t_s, gt_t_f = iterator.get_next()
    # Sub-series of ground truth series for initialization
    t_series_init = gt_t_series[:, :nbr_init_samples]

    # Construct RNN initialization graph
    t_preds_init, t_state_init = model.build_graph_init(cell, output_wrapper, t_series_init)
    t_preds_init = t_preds_init[-1:]

    # Create single step graph used to create samples
    # Separate input placeholder to feed back the predicted values as new input
    pl_x, pl_state, output, next_state = model.build_graph_single_step(cell, output_wrapper)

    sess.run(iterator.initializer)
    # Run init
    np_preds_init, np_state_cur, gt_np_s, gt_np_f, gt_np_series, gt_np_pred = sess.run([t_preds_init, t_state_init,
                                                                                        gt_t_s, gt_t_f, gt_t_series,
                                                                                        gt_t_pred])

    # Generate sin samples
    predictions = np_preds_init
    for i in range(nbr_samples - 1):
        feed = cell.create_state_feed_dict(pl_state, np_state_cur)
        # Feed back predicted value
        feed[pl_x] = predictions[i]
        np_pred_cur, np_state_cur = sess.run([output, next_state], feed_dict=feed)
        predictions.append(np_pred_cur)

    # Plot generated and ground truth wave
    pred_series = np.concatenate(predictions, axis=1)
    for i in range(5):
        x = (gt_np_s[i] + (nbr_init_samples - 1) * sample_step) + np.linspace(0, nbr_samples * sample_step,
                                                                              num=nbr_samples)
        plt.figure(i)
        plt.plot(x, np.squeeze(gt_np_pred[i, nbr_init_samples:]), 'ro', label='Ground Truth')
        plt.plot(x, np.squeeze(pred_series[i, :]), 'bo', label='Predicted')
        plt.legend()
        plt.savefig('Out_{}_{}.png'.format(rnn_type, i))


def run_memory_test(sess, loss, outputs, labels, it_test_series_init, to_remember_len):
    """ Run a memory task test on the given test series.

    :param sess: TensorFlow session
    :param loss: Loss used for training (TensorFlow tensor)
    :param outputs: Model outputs for each time step (already linearly mapped to the output dimension) (TensorFlow tensor)
    :param labels: Ground truth output (not one-hot encoded) (numpy array)
    :param it_test_series_init: Iterator for the test database
    :param to_remember_len: Size of the sequence to remember
    :return:
    """
    #####Start Subtask 1h#####
    outputs = tf.stack(outputs[-to_remember_len:], axis=1)
    acc, acc_update_op = tf.metrics.accuracy(labels[:, -to_remember_len:], tf.argmax(outputs, axis=2), name='test')
    sess.run([it_test_series_init, tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, 'test'))])

    nbr_batch = 0
    cum_loss = 0
    try:
        while True:
            cur_loss, _ = sess.run([loss, acc_update_op])
            nbr_batch += 1
            cum_loss += cur_loss
            print('Current test batch loss: %f' % cur_loss)
    except tf.errors.OutOfRangeError:
        logging.info('End of training')
        logging.info('Final average test batch loss: %f' % (cum_loss / nbr_batch))
        logging.info('Final memory accuracy: %f' % acc.eval())
    #####End Subtask#####
