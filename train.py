import os
import tensorflow as tf
from cnn import CNN
from config import CLASSES, LOG_DIR
from load_data import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 48, 'Image size.')
flags.DEFINE_integer('step_count', 1000, 'Number of steps.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


def main():
  with tf.Graph().as_default():
    cnn = CNN(image_size=FLAGS.image_size, class_count=len(CLASSES))
    images, labels = load_data(
      'data/train/data.csv',
      batch_size=FLAGS.batch_size,
      image_size=FLAGS.image_size,
      class_count=len(CLASSES),
      shuffle=True)
    keep_prob = tf.placeholder(tf.float32)

    logits = cnn.inference(images, keep_prob)
    loss = cnn.loss(logits, labels)
    train_op = cnn.training(loss, FLAGS.learning_rate)
    accuracy = cnn.accuracy(logits, labels)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

      for step in range(1, FLAGS.step_count + 1):
        _, loss_value, accuracy_value = sess.run(
          [train_op, loss, accuracy], feed_dict={keep_prob: 0.5})

        if step % 10 == 0:
          print(f'step {step}: training accuracy {accuracy_value}')
          summary = sess.run(summary_op, feed_dict={keep_prob: 1.0})
          summary_writer.add_summary(summary, step)

      coord.request_stop()
      coord.join(threads)

      save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))

if __name__ == '__main__':
  main()
