import argparse
import logging
import time
import train
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import input_cs
import model


def prepare_model(graph):
    """ Prepare the model graph

    :return: Input placeholder, Net linear output before softmax
    """

    with graph.as_default():
        # Placeholder for the images
        images = tf.placeholder(dtype=tf.float32, shape=[None] + input_cs.IMAGE_SIZE + [3], name='images')

        # Build model
        logits = model.build_model(images, input_cs.NUM_CLASSES)

    return images, logits


def test_model_file(sess, model_file, path_test):
    # Instantiate Saving and Restoring
    saver = tf.train.Saver()
    # Load model
    logging.info("Restoring model from %s" % model_file)
    saver.restore(sess, model_file)
    logging.info("Model restored")
    run_test(sess, images, logits, path_test)


def run_test(sess, images, logits, path_test):
    data_test, _ = input_cs.get_dataset_cs(path_test, 1, 128)
    data_it = data_test.make_one_shot_iterator()
    next_element = data_it.get_next()

    #####Insert your code here for subtask k#####
    # See the validation code snipped given on how to iterate over the test set


if __name__ == "__main__":
    #run this script e.g. like this
    #python3 main_cityscape.py --run_training --run_test --train_log_dir log/ --save_model_name models/model1

    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
      "--max_epochs",
      type=int,
      default=9,
      help="Number of epochs training is run.")
    parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="Batch size used during training.")
    parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.05,
      help="Initial learning rate.")
    parser.add_argument(
      "--data_dir_tr",
      type=str,
      default="cityscapesExtractedResized",
      help="Directory of the training data.")
    parser.add_argument(
      "--data_dir_val",
      type=str,
      default="cityscapesExtractedValResized",
      help="Directory of the validation data.")
    parser.add_argument(
      "--data_dir_test",
      type=str,
      default="cityscapesExtractedTestResized",
      help="Directory of the test data.")
    parser.add_argument(
      "--debug",
      type="bool",
      nargs='?',
      const=True,
      default=False,
      help="Use debugger to track down bad values during training.")
    parser.add_argument(
        "--validate_every",
        type=int,
        nargs=1,
        default=3,
        help="Run validation every x epochs.")
    parser.add_argument(
        "--run_training",
        type="bool",
        nargs='?',
        const=True,
        default=False,
        help="Training loop is run")
    parser.add_argument(
        "--run_test",
        type="bool",
        nargs='?',
        const=True,
        default=False,
        help="Run testing (after training if training is demanded")
    parser.add_argument(
        "--save_model_name",
        type=str,
        default=None,
        help="File the model is saved to")
    parser.add_argument(
        "--restore_model",
        type=str,
        default=None,
        help="File the model is restore from")
    parser.add_argument(
        "--train_log_dir",
        type=str,
        default=None,
        help="Directory for training logs")
    FLAGS, unparsed = parser.parse_known_args()
    validate_every = FLAGS.validate_every if type(FLAGS.validate_every) is int else FLAGS.validate_every[0]

    g = tf.Graph()
    images, logits = prepare_model(g)
    with g.as_default():
        with tf.Session() as sess:
            # Start tensorflow debug session
            if FLAGS.debug:
                logging.info("Start debug session")
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            if FLAGS.run_training:
                train.train(sess, FLAGS.data_dir_tr, FLAGS.data_dir_val, FLAGS.max_epochs, FLAGS.batch_size,
                            validate_every, FLAGS.learning_rate, g, images, logits,
                            model_file=None, save_name=FLAGS.save_model_name, train_log_dir=FLAGS.train_log_dir)

            if FLAGS.run_test:
                if FLAGS.run_training:
                    with g.as_default():
                        run_test(sess, images, logits, FLAGS.data_dir_test)
                else:
                    if FLAGS.restore_model:
                        with g.as_default():
                            test_model_file(sess, FLAGS.restore_model, FLAGS.data_dir_test)
                    else:
                        logging.error("Cannot test: No model trained or specified for restoring for testing")
            logging.info("Run time: %s" % (time.time() - start_time))
