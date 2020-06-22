import tensorflow as tf
import input_cs
import numpy as np

#####Start Solution#####
_kernel_sizes = [[5, 5, 3, 24], [5, 5, 24, 32], [5, 5, 32, 50]]  # List of conv kernel sizes. Data_format=(H,W,in,out)
_conv_strides = [[1, 1, 1, 1]] * len(_kernel_sizes)  # List of strides (all all dimensions) for each kernel
_pool_spec = {1: ([1, 3, 3, 1], [1, 2, 2, 1]),  # Dict of pooling kernels (size, stride). Data_format=(N,H,W,C)
              2: ([1, 3, 3, 1], [1, 2, 2, 1]),
              3: ([1, 3, 3, 1], [1, 2, 2, 1])}
_fcl_sizes = [100, 50]  # Sizes of fully connected layers


def get_conv_build_block(conv_kernel_size, conv_stride, pool_ker_size, pool_ker_stride, layer_in, block_num):
    """Add a basic building block (conv + relu + pool) to the current graph

    :param conv_kernel_size: 4 dim kernel size of convolution [batch (irrelevant), x, y, depth]
    :param conv_stride: Stride of the convolution
    :param pool_ker_size: 4 dim pooling window
    :param pool_ker_stride: Stride of the pooling
    :param layer_in: input tensor
    :param block_num: number of the block
    :return: Output tensor after convolution and pooling
    """
    # Add convolution layer
    conv_out = get_conv_layer(conv_kernel_size, conv_stride, layer_in, 'conv%d' % block_num)
    # Activation function
    act_out = tf.nn.relu(conv_out, name='post_activation_conv%d' % block_num)

    # Pooling
    pool_out = tf.nn.max_pool(act_out, ksize=pool_ker_size, strides=pool_ker_stride, padding='SAME',
                              name='pool%d' % block_num)

    return pool_out


def get_conv_layer(kernel_size, conv_stride, layer_in, name_scope):
    """Add convolutional layer to current graph

    :param kernel_size: 4d convolution kernel
    :param conv_stride: Stride of the convolution
    :param layer_in: Input tensor
    :param name_scope: Variable name scope for the convolution
    :return: Output tensor
    """

    with tf.variable_scope(name_scope) as scope:
        kernel_weights = tf.get_variable('weights', shape=kernel_size, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        conv = tf.nn.conv2d(layer_in, kernel_weights, conv_stride, padding='SAME')
        print("Model:", conv)
        biases = tf.get_variable('biases', kernel_size[3], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        layer_out = tf.nn.bias_add(conv, biases)

    return layer_out


def _get_fc_layer(n_in, n_out, layer_in, name_scope, applyReLU=True):
    """ Add fc layer with relu activation to current graph

    :param N_in: nbr input neurons
    :param n_out: Nbr output neurons
    :param layer_in: Input tensor
    :param name_scope: Variable name scope for layer
    :return: Output tensor
    """

    with tf.variable_scope(name_scope):
        weights = tf.get_variable('weights', shape=[n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[n_out], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))

    if applyReLU:
        layer_out = tf.nn.relu(tf.matmul(layer_in, weights) + biases, name='layer_out_act')
    else:
        layer_out = tf.add(tf.matmul(layer_in, weights), biases, name='layer_out')
    return layer_out
#####End Solution#####


def build_model(images, num_classes):
    """Build the model
    :param images: Input image placeholder
    :param num_classes: Nbr of final output classes
    :return: Output of final fc-layer
    """
    #####Start Subtask 1e#####
    # It might be useful to define helper functions which add a layer of type needed
    # If you define such as function, remember that multiple variables with the same name will result in an error
    # To this end you may want to use with tf.variable_scope(name) to define a named scope for each layer
    # This way, you get a less cluttered visualization of the graph in tensorboard and debugging may be easier in tfdbg

    # Convolutions and pooling
    layer_in = images  # layer_in corresponds to the images
    print("Model:", layer_in)
    for i, (kernel_sz, conv_stride) in enumerate(zip(_kernel_sizes, _conv_strides)):
        num_conv = i + 1

        if num_conv in _pool_spec:  # Do we want activation and pooling after convolution?
            pool_ker_sz, pool_ker_stride = _pool_spec[num_conv]
            layer_out = get_conv_build_block(kernel_sz, conv_stride,        # Add convolution, activation, pooling
                                             pool_ker_sz, pool_ker_stride,
                                             layer_in, num_conv)
        else:  # In case we want to add conv layers without pooling (not used at the moment)
            layer_out = get_conv_layer(kernel_sz, conv_stride, layer_in,  # Add only convolution
                                       'conv%d' % num_conv)
        layer_in = layer_out
        print("Model:", layer_in)
    #####End Subtask#####

    #####Start Subtask 1f#####
    # Add fc-classifictaion-layers
    with tf.variable_scope('fully_con_class') as scope_outer:
        shape_pool = layer_out.get_shape().as_list()  # size: [None, 8, 8, 50]
        dim = np.prod(shape_pool[1:])  # 8*8*50
        reshaped = tf.reshape(layer_out, [-1, dim])

        # Iterate over given fully connected layer sizes
        layer_in = reshaped
        dim_in = dim
        print("Model:", layer_in)
        for i, dim_out in enumerate(_fcl_sizes):
            layer_out = _get_fc_layer(dim_in, dim_out, layer_in, 'fc_layer_%d' % (i+1), True)
            layer_in = layer_out
            dim_in = dim_out
            print("Model:", layer_in)

        softmax_logits = _get_fc_layer(dim_in, num_classes, layer_out, 'softmax_linear', False)
        print("Model:", softmax_logits)
    #####End Subtask#####

    return softmax_logits


def loss(logits, labels):
    """ Add cross entropy loss to the graph
    :param logits: Linear logits for spare_softmax
    :param labels: Ground truth labels
    :return: Mean cross entropy loss, cross entropy loss for every training example seperately
            (used for validation purposes)
    """

    #####Start Subtask 1g#####
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,  # shape: (?,)
                                                                   logits=logits,  # shape: (?, 3)
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #####End Subtask#####

    return cross_entropy_mean, cross_entropy


def get_train_op_for_loss(loss, global_tr_step, initial_lr ):
    """Add training operation (gradient descent for given loss) to the graph

    :param loss: Loss value
    :param global_tr_step: Tensorflow global training step
    :param initial_lr: Initial learning rate
    :return: Training operation
    """

    #####Start Subtask 1h#####

    opt = tf.train.GradientDescentOptimizer(initial_lr)
    train_op = opt.minimize(loss=loss, global_step=global_tr_step)

    #####End Subtask#####

    return train_op


if __name__ == "__main__":
    images = tf.placeholder(dtype=tf.float32, shape=[None] + input_cs.IMAGE_SIZE + [3])
    logits = build_model(images, input_cs.NUM_CLASSES)
