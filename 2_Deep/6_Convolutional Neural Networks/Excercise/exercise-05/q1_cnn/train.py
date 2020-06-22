import model
import input_cs
import tensorflow as tf
import logging
import numpy as np




def train(sess, path_train, path_val, num_epochs, batch_size, validate_every, initial_lr,
          g, images, logits, model_file, save_name, train_log_dir):
    """Build and train the network

    Builds the model and starts training. Training via sgd with validation every validate_every steps. If save_file
    is specified, the model with the minimal validation loss is saved.

    :param sess: Session in which training should be run
    :param path_train: Path of training samples
    :param path_val: Path of validation samples
    :param num_epochs: Maximum number of epochs
    :param batch_size: Batch size
    :param validate_every: Validation every n epochs
    :param initial_lr: Initial learning rate
    :param g: the TensorFlow graph
    :param images: Images used as input to the network
    :param logits: network output (before softmax)
    :param model_file: Read model from file
    :param save_name: Save model to file
    :param train_log_dir:
    :return:
    """
    with g.as_default():
        # Get datasets
        data_train, data_train_filtered_names = input_cs.get_dataset_cs(path_train, num_epochs, batch_size)
        data_val, _ = input_cs.get_dataset_cs(path_val, num_epochs, 128)

        # --Get iterators for training and validation--
        # Handle for switching the iterator dataset
        handle = tf.placeholder(dtype=tf.string, shape=[])

        iterator = tf.contrib.data.Iterator.from_string_handle(
            handle, data_train.output_types, data_train.output_shapes)
        next_element = iterator.get_next()

        training_iterator = data_train.make_one_shot_iterator()
        validation_iterator = data_val.make_initializable_iterator()

        # Tensorflow global training step
        global_step = tf.train.get_or_create_global_step()

        # --Add loss and training operation--
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')
        cross_ent_mean, cross_ent = model.loss(logits, labels)
        train_op = model.get_train_op_for_loss(cross_ent_mean, global_step, batch_size, initial_lr )

        # Summaries
        merged_summary = tf.summary.merge_all()

        # Instantiate Saving and Restoring
        saver = tf.train.Saver()

        train_writer = None
        if train_log_dir:
            train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

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
                for i in range(input_cs.NUM_EX_TRAIN // batch_size):
                    # Run training and summary
                    #####Insert your code here for subtask 1i#####
                    # If you want to use the visualization in tensorboard you will need to run merged_summary
                    # at an appropriate place and assign the result to the summary variable
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
                    sess.run(validation_iterator.initializer)
                    try:
                        # Loop through validation set
                        while True:
                            #####Insert your code here for subtask 1j#####
                            # For predicting the label you may use the values output by evaluating: logits
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
