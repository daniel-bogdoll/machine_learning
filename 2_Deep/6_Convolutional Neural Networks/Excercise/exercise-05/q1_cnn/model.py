import tensorflow as tf
import input_cs
import numpy as np

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.



def build_model(images, num_classes):
    """Build the model

    :param images: Input image placeholder
    :param num_classes: Nbr of final output classes
    :return: Output of final fc-layer
    """
    #####Insert your code here for subtask 1e#####
    # It might be useful to define helper functions which add a layer of type needed
    # If you define such as function, remember that multiple variables with the same name will result in an error
    # To this end you may want to use with tf.variable_scope(name) to define a named scope for each layer
    # This way, you get a less cluttered visualization of the graph in tensorboard and debugging may be easier in tfdbg

    #####Insert your code here for subtask 1f#####
    # Add fc-classifictaion-layers

    return softmax_logits


def loss(logits, labels):
    """ Add cross entropy loss to the graph

    :param logits: Linear logits for spare_softmax
    :param labels: Ground truth labels
    :return: Mean cross entropy loss, cross entropy loss for every training example seperately
            (used for validation purposes)
    """

    #####Insert your code here for subtask 1g#####
    _add_loss_summaries(cross_entropy_mean)

    return cross_entropy_mean, cross_entropy


def get_train_op_for_loss(loss, global_tr_step, batch_size, initial_lr ):
    """Add training operation (gradient descent for given loss) to the graph

    :param loss: Loss value
    :param global_tr_step: Tensorflow global training step
    :param batch_size: Batch size
    :param initial_lr: Initial learning rate
    :return: Training operation
    """

    #####Insert your code here for subtask 1h#####
    return train_op


def _add_loss_summaries(loss):
    tf.summary.scalar('Loss', loss)


if __name__ == "__main__":
    images = tf.placeholder(dtype=tf.float32, shape=[None] + input_cs.IMAGE_SIZE + [3])
    logits = build_model(images, input_cs.NUM_CLASSES)
