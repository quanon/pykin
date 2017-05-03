import tensorflow as tf


class CNN:
  def __init__(self, image_size=48, class_count=2, color_channel_count=3):
    self.image_size = image_size
    self.class_count = class_count
    self.color_channel_count = color_channel_count

  def inference(self, x, keep_prob, softmax=False):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)

      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)

      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

    x_image = tf.reshape(
      x,
      [-1, self.image_size, self.image_size, self.color_channel_count])

    with tf.name_scope('conv1'):
      W_conv1 = weight_variable([5, 5, self.color_channel_count, 32])
      b_conv1 = bias_variable([32])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
      h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
      h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
      W_fc1 = weight_variable(
        [int(self.image_size / 4) * int(self.image_size / 4) * 64, 1024])
      b_fc1 = bias_variable([1024])
      h_pool2_flat = tf.reshape(
        h_pool2,
        [-1, int(self.image_size / 4) * int(self.image_size / 4) * 64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
      W_fc2 = weight_variable([1024, self.class_count])
      b_fc2 = bias_variable([self.class_count])
      y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    if softmax:
      with tf.name_scope('softmax'):
        y = tf.nn.softmax(y)

    return y

  def loss(self, y, labels):
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    tf.summary.scalar('cross_entropy', cross_entropy)

    return cross_entropy

  def training(self, cross_entropy, learning_rate=1e-4):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    return train_step

  def accuracy(self, y, labels):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    return accuracy
