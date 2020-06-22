import argparse
import tensorflow as tf
import MyLSTMCell
import BasicRNNCell
import logging
import RNNOutputWrapper
import data_generator
import model
import train
import test

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--task", type=int, default=1, help="Task number")
    parser.add_argument("--nbr_epochs", type=int, default=100, help="Number of epochs training is run")
    parser.add_argument("--sz_batch", type=int, default=500, help="Batch size used during training")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--sz_rnn", type=int, default=32, help="RNN size")
    parser.add_argument("--RNN", type=str, default="BasicRNN", help="RNN type to use: BasicRNN or LSTM")
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.task == 1:
        sz_dataset = 10000
        shape_data = 1
        nbr_test_samples = 100

        if FLAGS.RNN == 'BasicRNN':
            cell = BasicRNNCell.BasicRNNCell(shape_data, FLAGS.sz_rnn)
        elif FLAGS.RNN == 'LSTM':
            cell = MyLSTMCell.MyLSTMCell(shape_data, FLAGS.sz_rnn)
        else:
            raise Exception('Invalid RNN type given')

        rnn_out_wrapper = RNNOutputWrapper.RNNOutputWrapper(FLAGS.sz_rnn, shape_data)

        data_train = data_generator.generate_sin_data(sz_dataset, FLAGS.sz_batch)
        iterator = tf.contrib.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)

        train_next_element = iterator.get_next()
        train_series, train_label = train_next_element
        it_train_init = iterator.make_initializer(data_train)

        train_outputs = model.build_graph_train(cell, rnn_out_wrapper, train_series)
        train_loss = train.get_loss_sin(train_outputs, train_label)
        train_op = train.get_train_op(train_loss, FLAGS.lr)

        with tf.Session() as sess:
            train.run_training(sess, FLAGS.nbr_epochs, train_op, train_loss, it_train_init)
            test.run_test_sin(sess, cell, rnn_out_wrapper, nbr_test_samples, FLAGS.RNN)

    elif FLAGS.task == 2:
        sz_dataset = 10000
        db_size_valid = 1000
        db_size_test = 1000
        shape_letter = 10

        to_remember_len = 8
        blank_separation_len = 5

        if FLAGS.RNN == 'BasicRNN':
            cell = BasicRNNCell.BasicRNNCell(shape_letter, FLAGS.sz_rnn)
        elif FLAGS.RNN == 'LSTM':
            cell = MyLSTMCell.MyLSTMCell(shape_letter, FLAGS.sz_rnn)
        else:
            raise Exception('Invalid RNN type given')

        rnn_out_wrapper = RNNOutputWrapper.RNNOutputWrapper(FLAGS.sz_rnn, shape_letter)

        data_train = data_generator.generate_batch_memory_task(sz_dataset, FLAGS.sz_batch, to_remember_len,
                                                               blank_separation_len)
        data_test = data_generator.generate_batch_memory_task(db_size_test, FLAGS.sz_batch, to_remember_len,
                                                              blank_separation_len)
        iterator = tf.contrib.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
        next_element = iterator.get_next()
        series, label = next_element
        it_train_init = iterator.make_initializer(data_train)
        it_test_init = iterator.make_initializer(data_test)

        outputs = model.build_graph_train(cell, rnn_out_wrapper, series)
        loss = train.get_loss_memory(outputs, label)
        train_op = train.get_train_op(loss, FLAGS.lr)

        with tf.Session() as sess:
            train.run_training(sess, FLAGS.nbr_epochs, train_op, loss, it_train_init)
            test.run_memory_test(sess, loss, outputs, label, it_test_init, to_remember_len)

    # insert your code here

