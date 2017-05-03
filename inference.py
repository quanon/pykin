import numpy as np
import os
import sys
import tensorflow as tf
from cnn import CNN
from config import Channel, LOG_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 48, 'Image size.')


def load_image(imagepath, image_size):
  jpeg = tf.read_file(imagepath)
  image = tf.image.decode_jpeg(jpeg, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.image.resize_images(image, [image_size, image_size])
  image = tf.image.per_image_standardization(image)

  return image


def print_results(imagepath, softmax):
  os.system(f'imgcat {imagepath}')
  mex_channel_name_length = max(len(channel.name) for channel in Channel)
  for channel, value in zip(Channel, softmax):
    print(f'{channel.name.ljust(mex_channel_name_length + 1)}: {value}')

  print()

  prediction = Channel(np.argmax(softmax)).name
  for channel in Channel:
    if channel.name in imagepath:
      answer = channel.name
      break

  print(f'推測: {prediction}, 正解: {answer}')


def main(imagepath):
  cnn = CNN(image_size=FLAGS.image_size, class_count=len(Channel))
  image = load_image(imagepath, image_size=FLAGS.image_size)
  keep_prob = tf.placeholder(tf.float32)
  logits = cnn.inference(image, keep_prob, softmax=True)

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join(LOG_DIR, 'model.ckpt'))

  softmax = sess.run(logits, feed_dict={keep_prob: 1.0}).flatten()
  print_results(imagepath, softmax)

if __name__ == '__main__':
  main(sys.argv[1])
