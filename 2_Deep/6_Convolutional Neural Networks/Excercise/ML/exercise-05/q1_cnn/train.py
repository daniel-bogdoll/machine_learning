import model
import input_cs
import tensorflow as tf
import logging
import numpy as np


def train(sess, num_epochs, batch_size, validate_every, initial_lr, logits, labels, handle, training_handle,
          validation_handle, validation_initializer, model_file, save_name, train_log_dir):
    """Build and train the network

    Builds the model and starts training. Training via sgd with validation every validate_every steps. If save_file
    is specified, the model with the minimal validation loss is saved.

    :param sess: Session in which training should be run
    :param num_epochs: Maximum number of epochs
    :param batch_size: Batch size
    :param validate_every: Validation every n epochs
    :param initial_lr: Initial learning rate
    :param logits: network output (before softmax)
    :param labels: ground truth labels
    :param handle: placeholder for switching the iterator dataset
    :param training_handle: handle of training set which can be fed into "handle" to select the training set
    :param validation_handle: handle of validation set which can be fed into "handle" to select the validation set
    :param validation_initializer: an operation which needs to be run to initialize the validation set
    :param model_file: Read model from file
    :param save_name: Save model to file
    :param train_log_dir:
    :return:
    """
    # Tensorflow global training step
    global_step = tf.train.get_or_create_global_step()

    # --Add loss and training operation--
    cross_ent_mean, cross_ent = model.loss(logits, labels)
    train_op = model.get_train_op_for_loss(cross_ent_mean, global_step, initial_lr )

    tf.summary.scalar('Loss', cross_ent_mean)

    # Summaries
    merged_summary = tf.summary.merge_all()

    # Instantiate Saving and Restoring
    saver = tf.train.Saver()

    train_writer = None
    if train_log_dir:
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    # Load model or initialize fresh model
    if model_file:
        logging.info("Restoring model from %s" % model_file)
        saver.restore(sess, model_file)
        logging.info("Model restored")
        pass
    else:
        logging.info("Initializing variables")
        sess.run(tf.global_variables_initializer())

    min_val_loss = 10000
    try:
        # Start training loop
        # Run epoch
        for e in range(1, num_epochs + 1):
            logging.info("Run epoch %d" % e)
            # Approximately nbr steps in one epoch
            n_steps = input_cs.NUM_EX_TRAIN // batch_size
            for i in range(n_steps):
                # Run training and summary
                #####Start Subtask 1i#####
                # If you want to use the visualization in tensorboard you will need to run merged_summary
                # at an appropriate place and assign the result to the summary variable
                # use feed_dict={handle: training_handle} as an argument to sess.run in order to specify that
                # the training data should be used
                summary, _, curr_train_loss = sess.run([merged_summary, train_op, cross_ent_mean],
                                                       feed_dict={handle: training_handle})
                if i % 20 == 0:
                    logging.info("Train step: {}/{}, loss: {}".format((global_step.eval() - 1) % n_steps, n_steps,
                                                                      curr_train_loss))
                #####End Subtask#####
                if train_writer:
                    train_writer.add_summary(summary, global_step.eval())
            # Run validation
            if e % validate_every == 0:
                logging.info("Run validation")
                # Sum of cross entropy errors
                sum_ent_error = 0
                # Nbr of samples in validation set
                nbr_val_samples = 0
                # Nbr of correct predictions on validation set
                nbr_corr_pred = 0
                # Reset validation iterator
                sess.run(validation_initializer)
                try:
                    # Loop through validation set
                    while True:
                        #####Start Subtask 1j#####
                        out_cross_ent, out_logits, labels_val = sess.run(
                            [cross_ent, logits, labels], feed_dict={handle: validation_handle})

                        nbr_val_samples += labels_val.shape[0]
                        sum_ent_error += np.sum(out_cross_ent)
                        nbr_corr_pred += np.sum(np.argmax(out_logits, 1) == labels_val)
                        #####End Subtask#####
                except tf.errors.OutOfRangeError:
                    pass

                mean_val_loss = float(sum_ent_error) / nbr_val_samples
                val_accu = float(nbr_corr_pred) / nbr_val_samples
                # Log validation summary
                if train_writer:
                    val_summary = tf.Summary(value=[tf.Summary.Value(tag="validation_mean_ce",
                                                                     simple_value=mean_val_loss),
                                                    tf.Summary.Value(tag="validation_accuracy",
                                                                     simple_value=val_accu)])
                    train_writer.add_summary(val_summary, global_step.eval())
                    train_writer.flush()

                    logging.info("Validation mean: %f - accuracy: %f" % (mean_val_loss, val_accu))
                    if save_name:
                        if mean_val_loss < min_val_loss:
                            min_val_loss = mean_val_loss
                            saver.save(sess, save_name)
                            logging.info("Model saved in file: %s" % save_name)
    except tf.errors.OutOfRangeError:
        logging.info("End of training max epochs reached")
