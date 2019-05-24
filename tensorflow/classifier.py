import numpy as np
import tensorflow as tf
import math

import config


def variable_init_2d(num_input, num_output):
    """Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper.
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    init_tensor = tf.random.truncated_normal([int(num_input), int(num_output)], stddev=1.0 / math.sqrt(float(num_input)))
    return init_tensor


def build_ffnn(images, input_dim, hidden_list=[2000, 2000, 2000, 2000]):
    """ Build neural network model
    """
    hl = []
    for i in range(len(hidden_list)):
        if i == 0:
            Wh = tf.Variable(variable_init_2d(input_dim, hidden_list[i]), name='weights')
            bh = tf.Variable(tf.zeros([hidden_list[i]]), name='biases')
            hl.append(tf.nn.relu(tf.matmul(images, Wh) + bh))
        else:
            Wh = tf.Variable(variable_init_2d(hidden_list[i-1], hidden_list[i]), name='weights')
            bh = tf.Variable(tf.zeros([hidden_list[i]]), name='biases')
            hl.append(tf.nn.relu(tf.matmul(hl[i-1], Wh) + bh))

    return hl[len(hidden_list)-1]


def build_model(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(images, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h2 = tf.nn.relu(tf.matmul(out_h1_drop, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h3 = tf.nn.relu(tf.matmul(out_h2_drop, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h4 = tf.nn.relu(tf.matmul(out_h3_drop, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h4_drop, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, bh2, Wh3, bh3, Wh4, bh4, Wo, bo])
    return logits, saver


def build_model_nodrop(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(images, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h2 = tf.nn.relu(tf.matmul(out_h1, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h3 = tf.nn.relu(tf.matmul(out_h2, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        out_h4 = tf.nn.relu(tf.matmul(out_h3, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h2, Wo) + bo

    saver = tf.train.Saver()
    return logits, saver


def build_model2(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000, nodrop=True):
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(images, Wh1) + bh1)
        if nodrop is False:
            out_h1 = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h2 = tf.nn.relu(tf.matmul(out_h1, Wh2) + bh2)
        else:
            out_h2 = tf.nn.relu(tf.matmul(out_h1, Wh2) + bh2)
            out_h2 = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h3 = tf.nn.relu(tf.matmul(out_h2, Wh3) + bh3)
        else:
            out_h3 = tf.nn.relu(tf.matmul(out_h2, Wh3) + bh3)
            out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h4 = tf.nn.relu(tf.matmul(out_h3, Wh4) + bh4)
        else:
            out_h4 = tf.nn.relu(tf.matmul(out_h3, Wh4) + bh4)
            out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    with tf.name_scope('hidden5') as scope:
        Wh5 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh5 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h5 = tf.nn.relu(tf.matmul(out_h4, Wh5) + bh5)
        else:
            out_h5 = tf.nn.relu(tf.matmul(out_h4, Wh5) + bh5)
            out_h5_drop = tf.nn.dropout(out_h5, keep_prob)

    with tf.name_scope('hidden6') as scope:
        Wh6 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh6 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h6 = tf.nn.relu(tf.matmul(out_h5, Wh6) + bh6)
        else:
            out_h6 = tf.nn.relu(tf.matmul(out_h5, Wh6) + bh6)
            out_h6_drop = tf.nn.dropout(out_h6, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        if nodrop is True:
            logits = tf.matmul(out_h6, Wo) + bo
        else:
            logits = tf.matmul(out_h6_drop, Wo) + bo

    saver = tf.train.Saver()
    return logits, saver


def build_model2_reg(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000, nodrop=True):
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(images, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h2 = tf.nn.relu(tf.matmul(out_h1, Wh2) + bh2)
        else:
            out_h2 = tf.nn.relu(tf.matmul(out_h1_drop, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h3 = tf.nn.relu(tf.matmul(out_h2, Wh3) + bh3)
        else:
            out_h3 = tf.nn.relu(tf.matmul(out_h2_drop, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h4 = tf.nn.relu(tf.matmul(out_h3, Wh4) + bh4)
        else:
            out_h4 = tf.nn.relu(tf.matmul(out_h3_drop, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    with tf.name_scope('hidden5') as scope:
        Wh5 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh5 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h5 = tf.nn.relu(tf.matmul(out_h4, Wh5) + bh5)
        else:
            out_h5 = tf.nn.relu(tf.matmul(out_h4_drop, Wh5) + bh5)
        out_h5_drop = tf.nn.dropout(out_h5, keep_prob)

    with tf.name_scope('hidden6') as scope:
        Wh6 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh6 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        if nodrop is True:
            out_h6 = tf.nn.relu(tf.matmul(out_h5, Wh6) + bh6)
        else:
            out_h6 = tf.nn.relu(tf.matmul(out_h5_drop, Wh6) + bh6)
        out_h6_drop = tf.nn.dropout(out_h6, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        if nodrop is True:
            logits = tf.matmul(out_h6, Wo) + bo
        else:
            logits = tf.matmul(out_h6_drop, Wo) + bo

    reg = tf.nn.l2_loss(Wh1) + tf.nn.l2_loss(Wh2) + tf.nn.l2_loss(Wh3) + \
          tf.nn.l2_loss(Wh4) + tf.nn.l2_loss(Wh5) + tf.nn.l2_loss(Wh6)
    saver = tf.train.Saver()
    return logits, saver, reg


def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size,
                                             activation_fn=None,
                                             scope=scope)


def dense_batch_relu(x, phase, size, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


def build_model2_batch(images, output_dim, is_training=True):
    """ Build neural network model
    """

    h1 = dense_batch_relu(images, is_training, 5000, 'layer1')
    h2 = dense_batch_relu(h1, is_training, 5000, 'layer2')
    h3 = dense_batch_relu(h2, is_training, 5000, 'layer3')
    h4 = dense_batch_relu(h3, is_training, 5000, 'layer4')
    h5 = dense_batch_relu(h4, is_training, 5000, 'layer5')
    h6 = dense_batch_relu(h5, is_training, 5000, 'layer6')

    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(5000, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(h6, Wo) + bo

    saver = tf.train.Saver()
    return logits, saver


def init_weights(shape, name=None):

    if name is None:
        v = tf.Variable(tf.random_normal(shape, stddev=0.01))
    else:
        v = tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
    return v


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_model_cnn(images, keep_prob, batch_size, output_dim):

    x_image = tf.to_float(tf.reshape(images, [batch_size, 40, 40, 1]))

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([10 * 10 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, output_dim])
    b_fc2 = bias_variable([output_dim])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
    return y_conv, saver


def build_model_cnn2(images, keep_prob, batch_size, output_dim):
        x_image = tf.reshape(images, [batch_size, 40, 40, 1])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

        W_fc1 = weight_variable([5 * 5 * 128, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 5 * 5 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, output_dim])
        b_fc2 = bias_variable([output_dim])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
        return y_conv, saver


def build_model_cnn3(images, keep_prob, batch_size, output_dim):
    x_image = tf.reshape(images, [batch_size, 40, 40, 1])

    with tf.variable_scope('conv1') as scope:
        W_conv1 = _variable_with_weight_decay('weights',
                                              shape=[5, 5, 1, 16],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv1 = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
    with tf.variable_scope('conv1_1') as scope:
        W_conv1_1 = _variable_with_weight_decay('weights',
                                                shape=[5, 5, 16, 32],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv1_1 = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1, name=scope.name)
    h_pool1 = max_pool_2x2(h_conv1_1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = _variable_with_weight_decay('weights',
                                              shape=[4, 4, 32, 64],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2, name=scope.name)
    with tf.variable_scope('conv2_1') as scope:
        W_conv2_1 = _variable_with_weight_decay('weights',
                                                shape=[4, 4, 64, 128],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv2_1 = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1, name=scope.name)
    h_pool2 = max_pool_2x2(h_conv2_1)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    with tf.variable_scope('conv3') as scope:
        W_conv3 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 128, 256],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv3 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3, name=scope.name)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    with tf.variable_scope('fc1') as scope:
        W_fc1 = _variable_with_weight_decay('weights',
                                            shape=[5 * 5 * 256, 2048],
                                            stddev=0.04, wd=0.004)
        b_fc1 = _variable_on_cpu('biases', [2048], tf.constant_initializer(0.1))
        h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 5 * 5 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name=scope.name)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        W_fc2 = _variable_with_weight_decay('weights',
                                            shape=[2048, 1024],
                                            stddev=0.04, wd=0.004)
        b_fc2 = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name=scope.name)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('softmax_linear') as scope:
        W_fc3 = _variable_with_weight_decay('weights',
                                            shape=[1024, output_dim],
                                            stddev=0.04, wd=0.004)
        b_fc3 = _variable_on_cpu('biases', [output_dim], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name=scope.name)

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv1_1, b_conv1_1, W_conv2, b_conv2, W_conv2_1, b_conv2_1,
                            W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
    return softmax_linear, saver


def build_model_cnn3_1(images, keep_prob, batch_size, output_dim):
    # x_image = tf.to_float(tf.reshape(images, [batch_size, 40, 40, 1]))
    x_image = tf.reshape(images, [batch_size, 40, 40, 1])

    with tf.variable_scope('conv1') as scope:
        W_conv1 = _variable_with_weight_decay('weights',
                                              shape=[5, 5, 1, 32],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv1 = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
    with tf.variable_scope('conv1_1') as scope:
        W_conv1_1 = _variable_with_weight_decay('weights',
                                                shape=[5, 5, 32, 32],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv1_1 = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1, name=scope.name)
    h_pool1 = max_pool_2x2(h_conv1_1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = _variable_with_weight_decay('weights',
                                              shape=[4, 4, 32, 64],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2, name=scope.name)
    with tf.variable_scope('conv2_1') as scope:
        W_conv2_1 = _variable_with_weight_decay('weights',
                                                shape=[4, 4, 64, 64],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv2_1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1, name=scope.name)
    h_pool2 = max_pool_2x2(h_conv2_1)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    with tf.variable_scope('conv3') as scope:
        W_conv3 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 64, 128],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv3 = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3, name=scope.name)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    with tf.variable_scope('conv4') as scope:
        W_conv4 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 128, 256],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv4 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        h_conv4 = tf.nn.relu(conv2d(h_pool3_drop, W_conv4) + b_conv4, name=scope.name)
    h_pool4 = max_pool_2x2(h_conv4)
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob=keep_prob)

    with tf.variable_scope('fc1') as scope:
        W_fc1 = _variable_with_weight_decay('weights',
                                            shape=[3 * 3 * 256, 2048],
                                            stddev=0.04, wd=0.004)
        b_fc1 = _variable_on_cpu('biases', [2048], tf.constant_initializer(0.1))
        h_pool3_flat = tf.reshape(h_pool4_drop, [-1, 3 * 3 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name=scope.name)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        W_fc2 = _variable_with_weight_decay('weights',
                                            shape=[2048, 1024],
                                            stddev=0.04, wd=0.004)
        b_fc2 = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name=scope.name)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('softmax_linear') as scope:
        W_fc3 = _variable_with_weight_decay('weights',
                                            shape=[1024, output_dim],
                                            stddev=0.04, wd=0.004)
        b_fc3 = _variable_on_cpu('biases', [output_dim], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name=scope.name)

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv1_1, b_conv1_1, W_conv2, b_conv2, W_conv2_1, b_conv2_1,
                            W_conv3, b_conv3, W_conv4, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
    return softmax_linear, saver


def build_model_cnn3_2(images, keep_prob, batch_size, output_dim):
    x_image = tf.reshape(images, [batch_size, 40, 40, 1])

    with tf.variable_scope('conv1') as scope:
        W_conv1 = _variable_with_weight_decay('weights',
                                              shape=[5, 5, 1, 16],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv1 = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
    with tf.variable_scope('conv1_1') as scope:
        W_conv1_1 = _variable_with_weight_decay('weights',
                                                shape=[5, 5, 16, 32],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv1_1 = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1, name=scope.name)
    h_pool1 = max_pool_2x2(h_conv1_1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = _variable_with_weight_decay('weights',
                                              shape=[4, 4, 32, 64],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2, name=scope.name)
    with tf.variable_scope('conv2_1') as scope:
        W_conv2_1 = _variable_with_weight_decay('weights',
                                                shape=[4, 4, 64, 64],
                                                stddev=5e-2,
                                                wd=0.0)
        b_conv2_1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1, name=scope.name)
    h_pool2 = max_pool_2x2(h_conv2_1)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    with tf.variable_scope('conv3') as scope:
        W_conv3 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 64, 128],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv3 = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3, name=scope.name)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    with tf.variable_scope('conv4') as scope:
        W_conv4 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 128, 256],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv4 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        h_conv4 = tf.nn.relu(conv2d(h_pool3_drop, W_conv4) + b_conv4, name=scope.name)
    h_pool4 = max_pool_2x2(h_conv4)
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob=keep_prob)

    with tf.variable_scope('conv5') as scope:
        W_conv5 = _variable_with_weight_decay('weights',
                                              shape=[3, 3, 256, 256],
                                              stddev=5e-2,
                                              wd=0.0)
        b_conv5 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        h_conv5 = tf.nn.relu(conv2d(h_pool4_drop, W_conv5) + b_conv5, name=scope.name)
    h_pool5 = max_pool_2x2(h_conv5)
    h_pool5_drop = tf.nn.dropout(h_pool5, keep_prob=keep_prob)

    with tf.variable_scope('fc1') as scope:
        W_fc1 = _variable_with_weight_decay('weights',
                                            shape=[2 * 2 * 256, 2048],
                                            stddev=0.04, wd=0.004)
        b_fc1 = _variable_on_cpu('biases', [2048], tf.constant_initializer(0.1))
        h_pool3_flat = tf.reshape(h_pool5_drop, [-1, 2 * 2 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name=scope.name)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        W_fc2 = _variable_with_weight_decay('weights',
                                            shape=[2048, 1024],
                                            stddev=0.04, wd=0.004)
        b_fc2 = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name=scope.name)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('softmax_linear') as scope:
        W_fc3 = _variable_with_weight_decay('weights',
                                            shape=[1024, output_dim],
                                            stddev=0.04, wd=0.004)
        b_fc3 = _variable_on_cpu('biases', [output_dim], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name=scope.name)

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv1_1, b_conv1_1, W_conv2, b_conv2, W_conv2_1, b_conv2_1,
                            W_conv3, b_conv3, W_conv4, b_conv4, W_conv5, b_conv5,
                            W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
    return softmax_linear, saver


def build_model_cnn4(images, keep_prob, batch_size, output_dim):
    x_image = tf.reshape(images, [batch_size, 80, 80, 1])

    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    W_conv1_1 = weight_variable([5, 5, 16, 32])
    b_conv1_1 = bias_variable([32])
    h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1)
    h_pool1 = max_pool_2x2(h_conv1_1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    W_conv2_1 = weight_variable([4, 4, 64, 128])
    b_conv2_1 = bias_variable([128])
    h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)
    h_pool2 = max_pool_2x2(h_conv2_1)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    W_conv3 = weight_variable([3, 3, 128, 256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    W_fc1 = weight_variable([10 * 10 * 256, 2048])
    b_fc1 = bias_variable([2048])
    h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 10 * 10 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([2048, 1024])
    b_fc2 = bias_variable([1024])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([1024, output_dim])
    b_fc3 = bias_variable([output_dim])
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv1_1, b_conv1_1, W_conv2, b_conv2, W_conv2_1, b_conv2_1,
                            W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
    return y_conv, saver


def build_model_cnn5(images, keep_prob, batch_size, output_dim):
    x_image = tf.reshape(images, [batch_size, 80, 80, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    W_fc1 = weight_variable([10 * 10 * 128, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 10 * 10 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, output_dim])
    b_fc2 = bias_variable([output_dim])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
    return y_conv, saver


def build_model_cnn6(images, keep_prob, batch_size, output_dim):
    x_image = tf.reshape(images, [batch_size, 80, 80, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    W_conv4 = weight_variable([5, 5, 128, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3_drop, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob=keep_prob)

    W_fc1 = weight_variable([5 * 5 * 128, 1024])
    b_fc1 = bias_variable([1024])
    h_pool4_flat = tf.reshape(h_pool4_drop, [-1, 5 * 5 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, output_dim])
    b_fc2 = bias_variable([output_dim])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4,
                            W_fc1, b_fc1, W_fc2, b_fc2])
    return y_conv, saver


def build_seqnet3(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    image1, image2, image3, image4 = tf.split(images, num_or_size_splits=4, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim/4, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(image1, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        # in_h2 = tf.concat([out_h1_drop, image2], 1)
        out_h2 = tf.nn.relu(tf.matmul(out_h1_drop, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_2+input_dim/4, hidden_1), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        in_h3 = tf.concat([out_h2_drop, image2], 1)
        out_h3 = tf.nn.relu(tf.matmul(in_h3, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        # in_h4 = tf.concat([out_h3_drop, image4], 1)
        out_h4 = tf.nn.relu(tf.matmul(out_h3_drop, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    with tf.name_scope('hidden5') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/4, hidden_1), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        in_h5 = tf.concat([out_h4_drop, image3], 1)
        out_h5 = tf.nn.relu(tf.matmul(in_h5, Wh4) + bh4)
        out_h5_drop = tf.nn.dropout(out_h5, keep_prob)

    with tf.name_scope('hidden6') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        # in_h6 = tf.concat([out_h5_drop, image6], 1)
        out_h6 = tf.nn.relu(tf.matmul(out_h5_drop, Wh4) + bh4)
        out_h6_drop = tf.nn.dropout(out_h6, keep_prob)

    with tf.name_scope('hidden7') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/4, hidden_1), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        in_h7 = tf.concat([out_h6_drop, image4], 1)
        out_h7 = tf.nn.relu(tf.matmul(in_h7, Wh4) + bh4)
        out_h7_drop = tf.nn.dropout(out_h7, keep_prob)

    with tf.name_scope('hidden8') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        # in_h8 = tf.concat([out_h7_drop, image8], 1)
        out_h8 = tf.nn.relu(tf.matmul(out_h7_drop, Wh4) + bh4)
        out_h8_drop = tf.nn.dropout(out_h8, keep_prob)


    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h8_drop, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, bh2, Wh3, bh3, Wh4, bh4, Wo, bo])
    return logits, saver


def build_seqnet2(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    image1, image2, image3, image4, image5, image6, image7, image8, image9, image10 = tf.split(images, num_or_size_splits=10, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim/10, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(image1, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1+input_dim/10, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h2 = tf.concat([out_h1_drop, image2], 1)
        out_h2 = tf.nn.relu(tf.matmul(in_h2, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h3 = tf.concat([out_h2_drop, image3], 1)
        out_h3 = tf.nn.relu(tf.matmul(in_h3, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h4 = tf.concat([out_h3_drop, image4], 1)
        out_h4 = tf.nn.relu(tf.matmul(in_h4, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    with tf.name_scope('hidden5') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h5 = tf.concat([out_h4_drop, image5], 1)
        out_h5 = tf.nn.relu(tf.matmul(in_h5, Wh4) + bh4)
        out_h5_drop = tf.nn.dropout(out_h5, keep_prob)

    with tf.name_scope('hidden6') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h6 = tf.concat([out_h5_drop, image6], 1)
        out_h6 = tf.nn.relu(tf.matmul(in_h6, Wh4) + bh4)
        out_h6_drop = tf.nn.dropout(out_h6, keep_prob)

    with tf.name_scope('hidden7') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h7 = tf.concat([out_h6_drop, image7], 1)
        out_h7 = tf.nn.relu(tf.matmul(in_h7, Wh4) + bh4)
        out_h7_drop = tf.nn.dropout(out_h7, keep_prob)

    with tf.name_scope('hidden8') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h8 = tf.concat([out_h7_drop, image8], 1)
        out_h8 = tf.nn.relu(tf.matmul(in_h8, Wh4) + bh4)
        out_h8_drop = tf.nn.dropout(out_h8, keep_prob)

    with tf.name_scope('hidden9') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h9 = tf.concat([out_h8_drop, image9], 1)
        out_h9 = tf.nn.relu(tf.matmul(in_h9, Wh4) + bh4)
        out_h9_drop = tf.nn.dropout(out_h9, keep_prob)

    with tf.name_scope('hidden10') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/10, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h10 = tf.concat([out_h9_drop, image10], 1)
        out_h10 = tf.nn.relu(tf.matmul(in_h10, Wh4) + bh4)
        out_h10_drop = tf.nn.dropout(out_h10, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h10_drop, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, bh2, Wh3, bh3, Wh4, bh4, Wo, bo])
    return logits, saver


def build_seqnet(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    image1, image2, image3, image4 = tf.split(images, num_or_size_splits=4, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim/4, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(image1, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(hidden_1+input_dim/4, hidden_2), name='weights')
        # Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h2 = tf.concat([out_h1_drop, image2], 1)
        out_h2 = tf.nn.relu(tf.matmul(in_h2, Wh2) + bh2)
        # out_h2 = tf.nn.relu(tf.matmul(out_h1_drop, Wh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(hidden_2+input_dim/4, hidden_2), name='weights')
        # Wh3 = tf.Variable(variable_init_2d(hidden_2, hidden_2), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h3 = tf.concat([out_h2_drop, image3], 1)
        out_h3 = tf.nn.relu(tf.matmul(in_h3, Wh3) + bh3)
        # out_h3 = tf.nn.relu(tf.matmul(out_h1_drop, Wh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(hidden_2+input_dim/4, hidden_2), name='weights')
        # Wh4 = tf.Variable(variable_init_2d(hidden_2, hidden_2), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_2]), name='biases')
        in_h4 = tf.concat([out_h3_drop, image4], 1)
        out_h4 = tf.nn.relu(tf.matmul(in_h4, Wh4) + bh4)
        # out_h4 = tf.nn.relu(tf.matmul(out_h1_drop, Wh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h4, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_2, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h4_drop, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, bh2, Wh3, bh3, Wh4, bh4, Wo, bo])
    return logits, saver


def build_seqnet_npsrnn(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    image1, image2, image3, image4 = tf.split(images, num_or_size_splits=4, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim/4, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(image1, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(input_dim/4, hidden_1), name='weights')
        Rh2 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        # Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        # a = tf.matmul(image2, Wh2)
        # b = tf.matmul(Rh2, out_h1)
        out_h2 = tf.nn.relu(tf.matmul(image2, Wh2) + tf.matmul(out_h1, Rh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(input_dim/4, hidden_1), name='weights')
        Rh3 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h3 = tf.nn.relu(tf.matmul(image3, Wh3) + tf.matmul(out_h2, Rh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(input_dim / 4, hidden_1), name='weights')
        Rh4 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h4 = tf.nn.relu(tf.matmul(image4, Wh4) + tf.matmul(out_h3, Rh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_1, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h4, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, Rh2, bh2, Wh3, Rh3, bh3, Wh4, Rh4, bh4, Wo, bo])
    return logits, saver


def build_seqnet_npsrnn8(images, keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000):
    image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    # X --> Hidden 1
    with tf.name_scope('hidden1') as scope:
        Wh1 = tf.Variable(variable_init_2d(input_dim/8, hidden_1), name='weights')
        bh1 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        out_h1 = tf.nn.relu(tf.matmul(image1, Wh1) + bh1)
        out_h1_drop = tf.nn.dropout(out_h1, keep_prob)

    # Hidden 1 --> Hidden 2
    with tf.name_scope('hidden2') as scope:
        Wh2 = tf.Variable(variable_init_2d(input_dim/8, hidden_1), name='weights')
        Rh2 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        # Wh2 = tf.Variable(variable_init_2d(hidden_1, hidden_2), name='weights')
        bh2 = tf.Variable(tf.zeros([hidden_1]), name='biases')
        # a = tf.matmul(image2, Wh2)
        # b = tf.matmul(Rh2, out_h1)
        out_h2 = tf.nn.relu(tf.matmul(image2, Wh2) + tf.matmul(out_h1, Rh2) + bh2)
        out_h2_drop = tf.nn.dropout(out_h2, keep_prob)

    with tf.name_scope('hidden3') as scope:
        Wh3 = tf.Variable(variable_init_2d(input_dim/8, hidden_1), name='weights')
        Rh3 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh3 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h3 = tf.nn.relu(tf.matmul(image3, Wh3) + tf.matmul(out_h2, Rh3) + bh3)
        out_h3_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh4 = tf.Variable(variable_init_2d(input_dim /8, hidden_1), name='weights')
        Rh4 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh4 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h4 = tf.nn.relu(tf.matmul(image4, Wh4) + tf.matmul(out_h3, Rh4) + bh4)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden5') as scope:
        Wh5 = tf.Variable(variable_init_2d(input_dim / 8, hidden_1), name='weights')
        Rh5 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh5 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h5 = tf.nn.relu(tf.matmul(image5, Wh5) + tf.matmul(out_h4, Rh5) + bh5)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden6') as scope:
        Wh6 = tf.Variable(variable_init_2d(input_dim / 8, hidden_1), name='weights')
        Rh6 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh6 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h6 = tf.nn.relu(tf.matmul(image6, Wh6) + tf.matmul(out_h5, Rh6) + bh6)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden7') as scope:
        Wh7 = tf.Variable(variable_init_2d(input_dim / 8, hidden_1), name='weights')
        Rh7 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh7 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h7 = tf.nn.relu(tf.matmul(image7, Wh7) + tf.matmul(out_h6, Rh7) + bh7)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    with tf.name_scope('hidden4') as scope:
        Wh8 = tf.Variable(variable_init_2d(input_dim / 8, hidden_1), name='weights')
        Rh8 = tf.Variable(variable_init_2d(hidden_1, hidden_1), name='weights')
        bh8 = tf.Variable(tf.zeros([hidden_1]), name='biases')

        out_h8 = tf.nn.relu(tf.matmul(image8, Wh8) + tf.matmul(out_h7, Rh8) + bh8)
        out_h4_drop = tf.nn.dropout(out_h3, keep_prob)

    # Hidden 2 --> Y
    with tf.name_scope('softmax') as scope:
        Wo = tf.Variable(variable_init_2d(hidden_1, output_dim), name='weights')
        bo = tf.Variable(tf.zeros([output_dim]), name='biases')
        logits = tf.matmul(out_h8, Wo) + bo

    saver = tf.train.Saver([Wh1, bh1, Wh2, Rh2, bh2, Wh3, Rh3, bh3, Wh4, Rh4, bh4, Wo, bo])
    return logits, saver


def build_npsrnn(images, keep_prob, nFrm, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    W = []
    U = []
    b = []
    h = []

    for i in range(nFrm):

        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            # W.append(None)
            U.append(Ut)
            b.append(bt)
            h.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(h[i-1], Wt)
            ht = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            W.append(Wt)
            U.append(Ut)
            b.append(bt)
            h.append(ht)

    V = tf.Variable(variable_init_2d(nhiddenList[nFrm-1], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    a = tf.matmul(h[nFrm-1], V)
    logits = tf.add(a, c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn(images, keep_prob, nFrm, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    for i in range(nFrm):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    V = tf.Variable(variable_init_2d(nhiddenList[nFrm - 1], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    a = tf.matmul(hl[nFrm - 1], V)
    logits = tf.add(a, c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_rnn(images, keep_prob, nFrm, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    Wt = tf.Variable(variable_init_2d(nhiddenList[0], nhiddenList[0]), name='weights')
    Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    bt = tf.Variable(tf.zeros([nhiddenList[0]]), name='biases')
    Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    bTt = tf.Variable(tf.zeros([nhiddenList[0]]), name='biases')

    for i in range(nFrm):
        if i == 0:
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            # Ul.append(Ut)
            # bl.append(bt)
            # hl.append(ht)
        else:
            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(ht, Wt)
            H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(ht, C))
            # Wl.append(Wt)
            # Ul.append(Ut)
            # Tl.append(Tt)
            # bl.append(bt)
            # hl.append(ht)

    V = tf.Variable(variable_init_2d(nhiddenList[nFrm - 1], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    a = tf.matmul(ht, V)
    logits = tf.add(a, c)
    saver = tf.train.Saver()

    return logits, saver


def build_npsrnn_bidirectional(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    br = []
    hr = []

    for i in range(left_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            ht = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            Wl.append(Wt)
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            ht = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            Wr.append(Wt)
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    # tf.matmul(hl[left_context-1], V)
    # tf.matmul(hr[right_context-1], W)
    # tf.matmul(imagelist[left_context], X)

    # a = tf.matmul(h[nFrm-1], V)
    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_npsrnn_bidirectional_feed(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Rl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Rr = []
    br = []
    hr = []

    for i in range(left_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            Rt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            # ht = tf.nn.relu(tf.add(tf.add(tf.matmul(imagelist[i], Ut), tf.matmul(imagelist[left_context], Rt)), bt))
            ht = tf.nn.tanh(tf.add(tf.add(tf.matmul(imagelist[i], Ut), tf.matmul(imagelist[left_context], Rt)), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            Rt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            RtXm = tf.matmul(imagelist[left_context], Rt)
            # ht = tf.nn.relu(tf.add(tf.add(tf.add(UtXt, Wtht), RtXm), bt))
            ht = tf.nn.tanh(tf.add(tf.add(tf.add(UtXt, Wtht), RtXm), bt))
            Wl.append(Wt)
            Ul.append(Ut)
            Rl.append(Rt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            Rt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            # ht = tf.nn.relu(tf.add(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), tf.matmul(imagelist[left_context], Rt)), bt))
            ht = tf.nn.tanh(tf.add(tf.add(tf.matmul(imagelist[nFrm - 1 - i], Ut), tf.matmul(imagelist[left_context], Rt)), bt))
            # ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            RtXm = tf.matmul(imagelist[left_context], Rt)
            # ht = tf.nn.relu(tf.add(tf.add(tf.add(UtXt, Wtht), RtXm), bt))
            ht = tf.nn.tanh(tf.add(tf.add(tf.add(UtXt, Wtht), RtXm), bt))
            # ht = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            Wr.append(Wt)
            Ur.append(Ut)
            Rr.append(Rt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    # tf.matmul(hl[left_context-1], V)
    # tf.matmul(hr[right_context-1], W)
    # tf.matmul(imagelist[left_context], X)

    # a = tf.matmul(h[nFrm-1], V)
    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    left_context = int(left_context)
    right_context = int(right_context)
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    for i in range(int(left_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            # H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(int(right_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            # H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[int(left_context-1)], nhiddenList[int(left_context)]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[int(left_context+1)], nhiddenList[int(left_context)]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[int(left_context)]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_drop(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    left_context = int(left_context)
    right_context = int(right_context)
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    for i in range(int(left_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            ht_drop = tf.nn.dropout(ht, keep_prob)
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht_drop)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            ht_drop = tf.nn.dropout(ht, keep_prob)
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht_drop)

    for i in range(int(right_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            ht_drop = tf.nn.dropout(ht, keep_prob)
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht_drop)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            ht_drop = tf.nn.dropout(ht, keep_prob)
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht_drop)

    X = tf.Variable(variable_init_2d(nhiddenList[int(left_context-1)], nhiddenList[int(left_context)]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[int(left_context+1)], nhiddenList[int(left_context)]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[int(left_context)]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []
    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(left_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed_concat(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []
    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(left_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    XhlYhrZx = tf.concat([tf.concat([tf.matmul(hl[left_context-1], X), tf.matmul(hr[right_context-1], Y)], 1), tf.matmul(imagelist[left_context], Z)], 1)

    V = tf.Variable(variable_init_2d(nhiddenList[left_context]*3, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    logits = tf.add(tf.matmul(XhlYhrZx, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed_multilayer(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    left_context = int(left_context)
    right_context = int(right_context)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []
    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(left_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)

    FFDIM = 500
    V1 = tf.Variable(variable_init_2d(nhiddenList[left_context], FFDIM), name='weights')
    c1 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V2 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c2 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V3 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c3 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V = tf.Variable(variable_init_2d(FFDIM, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    h1 = tf.nn.relu(tf.add(tf.matmul(XhlYhrZx, V1), c1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, V2), c2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, V3), c3))

    logits = tf.add(tf.matmul(h3, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed_concat_multilayer(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []
    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(int(left_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[i], Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(int(right_context)):
        if i == 0:
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(imagelist[nFrm-1-i], Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    # XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)
    XhlYhrZx = tf.concat([tf.concat([tf.matmul(hl[left_context - 1], X), tf.matmul(hr[right_context - 1], Y)], 1),
                          tf.matmul(imagelist[left_context], Z)], 1)

    FFDIM = 3000
    V1 = tf.Variable(variable_init_2d(nhiddenList[left_context]*3, FFDIM), name='weights')
    c1 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V2 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c2 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V3 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c3 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V = tf.Variable(variable_init_2d(FFDIM, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    h1 = tf.nn.relu(tf.add(tf.matmul(XhlYhrZx, V1), c1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, V2), c2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, V3), c3))

    logits = tf.add(tf.matmul(h3, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed_concat_ffnnfeat(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    hl = []
    hidden_list = [2000, 1000, 2000]
    ffnn_w1 = tf.Variable(variable_init_2d(nDim, hidden_list[0]), name='weights')
    ffnn_b1 = tf.Variable(tf.zeros([hidden_list[0]]), name='biases')
    ffnn_w2 = tf.Variable(variable_init_2d(hidden_list[0], hidden_list[1]), name='weights')
    ffnn_b2 = tf.Variable(tf.zeros([hidden_list[1]]), name='biases')
    ffnn_w3 = tf.Variable(variable_init_2d(hidden_list[1], hidden_list[2]), name='weights')
    ffnn_b3 = tf.Variable(tf.zeros([hidden_list[2]]), name='biases')




    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(left_context):
        if i == 0:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Ut = tf.Variable(variable_init_2d(hidden_list[len(hidden_list)-1], nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(ffnn_h3, Ut), bt))
            Ul.append(Ut)
            bl.append(bt)
            hl.append(ht)
        else:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            Ut = tf.Variable(variable_init_2d(hidden_list[len(hidden_list)-1], nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(ffnn_h3, Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Ul.append(Ut)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Ut = tf.Variable(variable_init_2d(hidden_list[len(hidden_list)-1], nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(ffnn_h3, Ut), bt))
            Ur.append(Ut)
            br.append(bt)
            hr.append(ht)
        else:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm - 1 - i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            Ut = tf.Variable(variable_init_2d(hidden_list[len(hidden_list)-1], nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(ffnn_h3, Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Ur.append(Ut)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    # XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)
    XhlYhrZx = tf.concat([tf.concat([tf.matmul(hl[left_context - 1], X), tf.matmul(hr[right_context - 1], Y)], 1),
                          tf.matmul(imagelist[left_context], Z)], 1)

    FFDIM = 3000
    V1 = tf.Variable(variable_init_2d(nhiddenList[left_context]*3, FFDIM), name='weights')
    c1 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V2 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c2 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V3 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c3 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V = tf.Variable(variable_init_2d(FFDIM, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    h1 = tf.nn.relu(tf.add(tf.matmul(XhlYhrZx, V1), c1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, V2), c2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, V3), c3))

    logits = tf.add(tf.matmul(h3, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_hn_npsrnn_bidirectional_feed_concat_ffnnfeat_sameUt(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    nFrm = len(nhiddenList)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    Wl = []
    Ul = []
    Tl = []
    bl = []
    hl = []
    Wr = []
    Ur = []
    Tr = []
    br = []
    hr = []

    hl = []
    hidden_list = [2000, 1000, 2000]
    ffnn_w1 = tf.Variable(variable_init_2d(nDim, hidden_list[0]), name='weights')
    ffnn_b1 = tf.Variable(tf.zeros([hidden_list[0]]), name='biases')
    ffnn_w2 = tf.Variable(variable_init_2d(hidden_list[0], hidden_list[1]), name='weights')
    ffnn_b2 = tf.Variable(tf.zeros([hidden_list[1]]), name='biases')
    ffnn_w3 = tf.Variable(variable_init_2d(hidden_list[1], hidden_list[2]), name='weights')
    ffnn_b3 = tf.Variable(tf.zeros([hidden_list[2]]), name='biases')
    Ut = tf.Variable(variable_init_2d(hidden_list[len(hidden_list) - 1], nhiddenList[0]), name='weights')




    Fd = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    FdX = tf.matmul(imagelist[left_context], Fd)

    for i in range(left_context):
        if i == 0:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(ffnn_h3, Ut), bt))
            bl.append(bt)
            hl.append(ht)
        else:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Wt = tf.Variable(variable_init_2d(nhiddenList[i-1], nhiddenList[i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[i]]), name='biases')

            UtXt = tf.matmul(ffnn_h3, Ut)
            Wtht = tf.matmul(hl[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hl[i-1], C))
            Wl.append(Wt)
            Tl.append(Tt)
            bl.append(bt)
            hl.append(ht)

    for i in range(right_context):
        if i == 0:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm-1-i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            ht = tf.nn.relu(tf.add(tf.matmul(ffnn_h3, Ut), bt))
            br.append(bt)
            hr.append(ht)
        else:
            ffnn_h1 = tf.nn.relu(tf.add(tf.matmul(imagelist[nFrm - 1 - i], ffnn_w1), ffnn_b1))
            ffnn_h2 = tf.nn.relu(tf.add(tf.matmul(ffnn_h1, ffnn_w2), ffnn_b2))
            ffnn_h3 = tf.nn.relu(tf.add(tf.matmul(ffnn_h2, ffnn_w3), ffnn_b3))
            Wt = tf.Variable(variable_init_2d(nhiddenList[nFrm-i], nhiddenList[nFrm-1-i]), name='weights')
            bt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')
            Tt = tf.Variable(variable_init_2d(nDim, nhiddenList[nFrm-1-i]), name='weights')
            bTt = tf.Variable(tf.zeros([nhiddenList[nFrm-1-i]]), name='biases')

            UtXt = tf.matmul(ffnn_h3, Ut)
            Wtht = tf.matmul(hr[i-1], Wt)
            H = tf.nn.relu(tf.add(tf.add(UtXt, Wtht), bt))
            T = tf.sigmoid(tf.matmul(imagelist[nFrm-1-i], Tt) + FdX + bTt)
            C = tf.subtract(1.0, T)
            ht = tf.add(tf.multiply(H, T), tf.multiply(hr[i-1], C))
            Wr.append(Wt)
            Tr.append(Tt)
            br.append(bt)
            hr.append(ht)

    X = tf.Variable(variable_init_2d(nhiddenList[left_context-1], nhiddenList[left_context]), name='weights')
    Y = tf.Variable(variable_init_2d(nhiddenList[left_context+1], nhiddenList[left_context]), name='weights')
    Z = tf.Variable(variable_init_2d(nDim, nhiddenList[left_context]), name='weights')

    # XhlYhrZx = tf.matmul(hl[left_context-1], X) + tf.matmul(hr[right_context-1], Y) + tf.matmul(imagelist[left_context], Z)
    XhlYhrZx = tf.concat([tf.concat([tf.matmul(hl[left_context - 1], X), tf.matmul(hr[right_context - 1], Y)], 1),
                          tf.matmul(imagelist[left_context], Z)], 1)

    FFDIM = 3000
    V1 = tf.Variable(variable_init_2d(nhiddenList[left_context]*3, FFDIM), name='weights')
    c1 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V2 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c2 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V3 = tf.Variable(variable_init_2d(FFDIM, FFDIM), name='weights')
    c3 = tf.Variable(tf.zeros(FFDIM), name='biases')
    V = tf.Variable(variable_init_2d(FFDIM, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    h1 = tf.nn.relu(tf.add(tf.matmul(XhlYhrZx, V1), c1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, V2), c2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, V3), c3))

    logits = tf.add(tf.matmul(h3, V), c)
    saver = tf.train.Saver()

    return logits, saver


def build_RNN(images, keep_prob, left_context, right_context, nDim, output_dim, nhiddenList):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    timesteps = left_context + right_context + 1

    tf.reshape(images, [None, timesteps, nDim/timesteps])
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(images, timesteps, 1)

    # Define a lstm cell with tensorflow
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in nhiddenList]

    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(stacked_rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return outputs


def build_npsrnn_tiedparam(images, nFrm, nDim, output_dim, nhiddenList):
    # image1, image2, image3, image4, image5, image6, image7, image8 = tf.split(images, num_or_size_splits=8, axis=1)
    imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
    # x_image = tf.reshape(images, [batch_size, 40, 40, 1])
    """ Build neural network model
    """
    W = []
    U = []
    b = []
    h = []
    Ut = tf.Variable(variable_init_2d(nDim, nhiddenList[0]), name='weights')
    Wt = tf.Variable(variable_init_2d(nhiddenList[0], nhiddenList[0]), name='weights')
    bt = tf.Variable(tf.zeros([nhiddenList[0]]), name='biases')
    for i in range(nFrm):

        if i == 0:
            ht = tf.nn.tanh(tf.add(tf.matmul(imagelist[i], Ut), bt))
            # W.append(None)
            U.append(Ut)
            b.append(bt)
            h.append(ht)
        else:
            UtXt = tf.matmul(imagelist[i], Ut)
            Wtht = tf.matmul(h[i-1], Wt)
            ht = tf.nn.tanh(tf.add(tf.add(UtXt, Wtht), bt))
            W.append(Wt)
            U.append(Ut)
            b.append(bt)
            h.append(ht)

    V = tf.Variable(variable_init_2d(nhiddenList[nFrm-1], output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    a = tf.matmul(h[nFrm-1], V)
    logits = tf.add(a, c)
    saver = tf.train.Saver()

    return logits, saver


def build_lstm_rnn(images, context_length, nDim, output_dim, nhidden):

    imagelist = tf.split(images, num_or_size_splits=context_length, axis=1)

    W_ig_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_ig_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_ig_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_ig = tf.Variable(tf.zeros([nhidden]), name='biases')
    W_og_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_og_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_og_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_og = tf.Variable(tf.zeros([nhidden]), name='biases')
    W_fg_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_fg_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_fg_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_fg = tf.Variable(tf.zeros([nhidden]), name='biases')
    W_mc_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_mc_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_mc = tf.Variable(tf.zeros([nhidden]), name='biases')

    # ct = []
    # ht = []
    ct = None
    ht = None
    for i in range(context_length):
        if i == 0:
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + b_og)
            # ct.append(tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x)+b_mc)))
            # ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
            ct = tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x)+b_mc))
            ht = tf.multiply(og, tf.nn.tanh(ct))
        else:
            print(i)
            # ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + tf.matmul(ht[i-1], W_ig_h) + tf.matmul(ct[i-1], W_ig_c) + b_ig)
            # og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + tf.matmul(ht[i-1], W_og_h) + tf.matmul(ct[i-1], W_og_c) + b_og)
            # fg = tf.nn.sigmoid(tf.matmul(imagelist[i], W_fg_x) + tf.matmul(ht[i-1], W_fg_h) + tf.matmul(ct[i-1], W_fg_c) + b_fg)
            # ct.append(tf.multiply(fg, ct[i-1]) + tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x) + tf.matmul(ht[i-1], W_mc_h) + b_mc)))
            # ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + tf.matmul(ht, W_ig_h) + tf.matmul(ct, W_ig_c) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + tf.matmul(ht, W_og_h) + tf.matmul(ct, W_og_c) + b_og)
            fg = tf.nn.sigmoid(tf.matmul(imagelist[i], W_fg_x) + tf.matmul(ht, W_fg_h) + tf.matmul(ct, W_fg_c) + b_fg)
            ct = tf.multiply(fg, ct) + tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x) + tf.matmul(ht, W_mc_h) + b_mc))
            ht = tf.multiply(og, tf.nn.tanh(ct))

    V = tf.Variable(variable_init_2d(nhidden, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    saver = tf.train.Saver()
    # return tf.matmul(ht[context_length-1], V) + c, saver
    return tf.matmul(ht, V) + c, saver


def build_pnslstm_rnn(images, context_length, nDim, output_dim, nhidden):

    imagelist = tf.split(images, num_or_size_splits=context_length, axis=1)

    W_ig_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_ig_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_ig_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_ig = tf.Variable(tf.zeros([nhidden]), name='biases')
    W_og_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_og_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_og_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_og = tf.Variable(tf.zeros([nhidden]), name='biases')
    W_fg_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
    W_fg_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    W_fg_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
    b_fg = tf.Variable(tf.zeros([nhidden]), name='biases')


    ct = []
    ht = []
    W_mc_xt = []
    W_mc_ht = []
    b_mct = []
    for i in range(context_length):
        W_mc_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
        W_mc_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        b_mc = tf.Variable(tf.zeros([nhidden]), name='biases')
        if i == 0:
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + b_og)
            ct.append(tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x)+b_mc)))
            ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
        else:
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + tf.matmul(ht[i-1], W_ig_h) + tf.matmul(ct[i-1], W_ig_c) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + tf.matmul(ht[i-1], W_og_h) + tf.matmul(ct[i-1], W_og_c) + b_og)
            fg = tf.nn.sigmoid(tf.matmul(imagelist[i], W_fg_x) + tf.matmul(ht[i-1], W_fg_h) + tf.matmul(ct[i-1], W_fg_c) + b_fg)
            ct.append(tf.multiply(fg, ct[i-1]) + tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x) + tf.matmul(ht[i-1], W_mc_h) + b_mc)))
            ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
        W_mc_xt.append(W_mc_x)
        W_mc_ht.append(W_mc_ht)
        b_mct.append(b_mc)

    V = tf.Variable(variable_init_2d(nhidden, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    saver = tf.train.Saver()
    return tf.matmul(ht[context_length-1], V) + c, saver


def build_fpnslstm_rnn(images, context_length, nDim, output_dim, nhidden):

    imagelist = tf.split(images, num_or_size_splits=context_length, axis=1)




    ct = []
    ht = []
    W_mc_xt = []
    W_mc_ht = []
    b_mct = []
    for i in range(context_length):
        W_ig_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
        W_ig_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        W_ig_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        b_ig = tf.Variable(tf.zeros([nhidden]), name='biases')
        W_og_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
        W_og_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        W_og_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        b_og = tf.Variable(tf.zeros([nhidden]), name='biases')
        W_fg_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
        W_fg_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        W_fg_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        b_fg = tf.Variable(tf.zeros([nhidden]), name='biases')
        W_mc_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
        W_mc_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
        b_mc = tf.Variable(tf.zeros([nhidden]), name='biases')
        if i == 0:
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + b_og)
            ct.append(tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x)+b_mc)))
            ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
        else:
            ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + tf.matmul(ht[i-1], W_ig_h) + tf.matmul(ct[i-1], W_ig_c) + b_ig)
            og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + tf.matmul(ht[i-1], W_og_h) + tf.matmul(ct[i-1], W_og_c) + b_og)
            fg = tf.nn.sigmoid(tf.matmul(imagelist[i], W_fg_x) + tf.matmul(ht[i-1], W_fg_h) + tf.matmul(ct[i-1], W_fg_c) + b_fg)
            ct.append(tf.multiply(fg, ct[i-1]) + tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x) + tf.matmul(ht[i-1], W_mc_h) + b_mc)))
            ht.append(tf.multiply(og, tf.nn.tanh(ct[i])))
        W_mc_xt.append(W_mc_x)
        W_mc_ht.append(W_mc_ht)
        b_mct.append(b_mc)

    V = tf.Variable(variable_init_2d(nhidden, output_dim), name='weights')
    c = tf.Variable(tf.zeros(output_dim), name='biases')

    saver = tf.train.Saver()
    return tf.matmul(ht[context_length-1], V) + c, saver


# def build_pnslstm_rnn_bidirection(images, left_context, right_context,, nDim, output_dim, nhidden):
#     nFrm = left_context + right_context + 1
#     imagelist = tf.split(images, num_or_size_splits=nFrm, axis=1)
#
#     W_ig_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
#     W_ig_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     W_ig_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     b_ig = tf.Variable(tf.zeros([nhidden]), name='biases')
#     W_og_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
#     W_og_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     W_og_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     b_og = tf.Variable(tf.zeros([nhidden]), name='biases')
#     W_fg_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
#     W_fg_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     W_fg_c = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     b_fg = tf.Variable(tf.zeros([nhidden]), name='biases')
#     W_mc_x = tf.Variable(variable_init_2d(nDim, nhidden), name='weights')
#     W_mc_h = tf.Variable(variable_init_2d(nhidden, nhidden), name='weights')
#     b_mc = tf.Variable(tf.zeros([nhidden]), name='biases')
#
#     cl = []
#     hl = []
#     cr = []
#     hr = []
#     for i in range(left_context):
#         if i == 0:
#             ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + b_ig)
#             og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + b_og)
#             cl.append(tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x)+b_mc)))
#             hl.append(tf.multiply(og, tf.nn.tanh(cl[i])))
#         else:
#             ig = tf.nn.sigmoid(tf.matmul(imagelist[i], W_ig_x) + tf.matmul(hl[i-1], W_ig_h) + tf.matmul(cl[i-1], W_ig_c) + b_ig)
#             og = tf.nn.sigmoid(tf.matmul(imagelist[i], W_og_x) + tf.matmul(hl[i-1], W_og_h) + tf.matmul(cl[i-1], W_og_c) + b_og)
#             fg = tf.nn.sigmoid(tf.matmul(imagelist[i], W_fg_x) + tf.matmul(hl[i-1], W_fg_h) + tf.matmul(cl[i-1], W_fg_c) + b_fg)
#             cl.append(tf.multiply(fg, cl[i-1]) + tf.multiply(ig, (tf.matmul(imagelist[i], W_mc_x) + tf.matmul(hl[i-1], W_mc_h) + b_mc)))
#             hl.append(tf.multiply(og, tf.nn.tanh(cl[i])))
#
#     V = tf.Variable(variable_init_2d(nhidden, output_dim), name='weights')
#     c = tf.Variable(tf.zeros(output_dim), name='biases')
#
#     saver = tf.train.Saver()
#     return tf.matmul(ht[context_length-1], V) + c, saver


def cross_entropy_loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels,
                                                            name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def check_accuracy(sess, eval_op, pl_images, pl_labels, pl_keep_prob, images, labels, batch_size, device_id):
    n_batches = images.shape[0] / batch_size
    true_counts = []
    num_examples = 0
    for i in xrange(n_batches-1):
        feed_dict = {
            pl_images:images[i*batch_size:(i+1)*batch_size],
            pl_labels:labels[i*batch_size:(i+1)*batch_size],
            pl_keep_prob:1.0,
        }
        with tf.device(device_id):
            true_count = sess.run(eval_op, feed_dict=feed_dict)
        true_counts.append(true_count)
        num_examples += batch_size

    return num_examples, true_counts


def classify(sess, softmax, predict_op, batch_size, device_id, pl_images, pl_keep_prob, images):

    frame_result = []
    frame_result_prob = None
    frame_result_prob_total = np.zeros(config.NUM_CLASS)
    class_count = np.zeros(config.NUM_CLASS, np.int32)
    n_batches = images.shape[0] / batch_size

    for i in xrange(n_batches):
        feed_dict = {
            pl_images:images[i*batch_size:(i+1)*batch_size],
            # pl_labels:labels[i*batch_size:(i+1)*batch_size],
            pl_keep_prob:1.0
        }
        with tf.device(device_id):
            result_predict_op = sess.run(predict_op, feed_dict=feed_dict)
            result_softmax = sess.run(softmax, feed_dict=feed_dict)

            if i == 0:
                frame_result_prob = np.asarray(result_softmax)
            else:
                frame_result_prob = np.append(frame_result_prob, np.asarray(result_softmax), axis=0)

            # print 'res'
            # print result_softmax
            for idx, value in enumerate(result_predict_op):
                frame_result.append(value)
                class_count[value] += 1
            for idx, value in enumerate(result_softmax):
                frame_result_prob_total += value

    return frame_result, class_count, frame_result_prob, frame_result_prob_total


def analysis(sess, softmax, pl_images, pl_labels, pl_keep_prob, images, labels, batch_size, device_id):
    n_batches = images.shape[0] / batch_size
    sfs = None
    num_examples = 0
    # for i in xrange(10):
    for i in xrange(n_batches - 1):
        feed_dict = {
            pl_images:images[i*batch_size:(i+1)*batch_size],
            pl_labels:labels[i*batch_size:(i+1)*batch_size],
            pl_keep_prob:1.0,
        }
        curlabels = labels[i*batch_size:(i+1)*batch_size]
        with tf.device(device_id):
            sf = sess.run(softmax, feed_dict=feed_dict)
        argmax = np.argmax(sf, axis=1)
        eq = np.equal(argmax, curlabels)
        for j in range(batch_size):
            if eq[j] == 0:
                print (curlabels[j])
        if i == 0:
            sfs = np.amax(sf, axis=1)
        else:
            sfs = np.append(sfs, np.amax(sf, axis=1))
        num_examples += batch_size

    return num_examples, sfs