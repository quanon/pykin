import os
import tensorflow as tf
from cnn import CNN
from config import Channel, LOG_DIR
from load_data import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 48, 'Image size.')
flags.DEFINE_integer('batch_size', 1000, 'Batch size.')


def main():
  with tf.Graph().as_default():
    cnn = CNN(image_size=FLAGS.image_size, class_count=len(Channel))
    images, labels = load_data(
      'data/test/data.csv',
      batch_size=FLAGS.batch_size,
      image_size=FLAGS.image_size,
      class_count=len(Channel),
      shuffle=False)
    keep_prob = tf.placeholder(tf.float32)

    logits = cnn.inference(images, keep_prob)
    accuracy = cnn.accuracy(logits, labels)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init_op)
      saver.restore(sess, os.path.join(LOG_DIR, 'model.ckpt'))
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      accuracy_value = sess.run(accuracy, feed_dict={keep_prob: 0.5})

      print(f'test accuracy: {accuracy_value}')

      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  main()
