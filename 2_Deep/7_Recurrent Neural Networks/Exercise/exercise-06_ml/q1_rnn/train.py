import tensorflow as tf
import logging


def run_training(sess, nbr_epochs, train_op, total_loss, it_train_init):
    """

    :param sess: TensorFlow session used for training
    :param nbr_epochs: Number of training epochs
    :param train_op: Training step operation
    :param total_loss: Batch loss
    :param it_train_init: Training dataset iterator initializer
    :return:
    """

    sess.run(tf.global_variables_initializer())
    # Run training epochs
    for e in range(nbr_epochs):
        sess.run(it_train_init)
        loss_acc = 0.0
        n_batch = 0
        try:
            while True:
                _, cur_loss = sess.run([train_op, total_loss])
                n_batch += 1
                loss_acc += cur_loss
                if n_batch % 50 == 0:
                    logging.info('Epoch %d: Train batch %d loss: %f' % (e, n_batch, cur_loss))
        except tf.errors.OutOfRangeError:
            logging.info('End of epoch %d: Avg loss %f' % (e, loss_acc / n_batch))


def get_loss_sin(outputs, y):
    """ Loss for the sin learning task. Mean squared error loss is used

    :param outputs: Model outputs
    :param y: Expected outputs
    :return: Mean squared error
    """

    #####Start Subtask 1e#####

    outputs = tf.stack(outputs, axis=1)
    loss = tf.losses.mean_squared_error(y, tf.squeeze(outputs, axis=2))

    #####End Subtask#####

    return loss


def get_train_op(loss, lr):
    """ Training operation is simply gradient descent with momentum.

    :param loss: Loss tensor
    :param lr: Learning rate
    :return:
    """

    #####Start Subtask 1f#####

    #optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    optimizer = tf.train.AdamOptimizer(lr)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, _ = tf.clip_by_global_norm(grads, 5.0)  # perform clipping to avoid exploding gradients
    train_op = optimizer.apply_gradients(zip(grads, variables))

    #####End Subtask#####

    return train_op


def get_loss_memory(outputs, y):
    """ Get loss for the memory task.

    :param outputs: Model outputs
    :param y: Expected labels
    :return: Batch loss
    """
    #####Start Subtask 1g#####
    outputs = tf.stack(outputs, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(y, outputs)
    #####End Subtask#####

    return loss
