import tensorflow as tf


def load_data(csvpath, batch_size, image_size, class_count,
  shuffle=False, min_after_dequeue=1000):

  queue = tf.train.string_input_producer([csvpath], shuffle=shuffle)
  reader = tf.TextLineReader()
  key, value = reader.read(queue)
  imagepath, label = tf.decode_csv(value, [['imagepath'], [0]])

  jpeg = tf.read_file(imagepath)
  image = tf.image.decode_jpeg(jpeg, channels=3)
  image = tf.image.resize_images(image, [image_size, image_size])
  image = tf.image.per_image_standardization(image)

  label = tf.one_hot(label, depth=class_count, dtype=tf.float32)

  capacity = min_after_dequeue + batch_size * 3

  if shuffle:
    images, labels = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=4,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  else:
    images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      capacity=capacity)

  return images, labels
