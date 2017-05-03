import glob
import numpy as np
import os
import shutil


def clean_data():
  for dirpath, _, filenames in os.walk('data'):
    for filename in filenames:
      os.remove(os.path.join(dirpath, filename))


def split_pathnames(dirpath):
  pathnames = glob.glob(f'{dirpath}/*')
  np.random.shuffle(pathnames)

  return np.split(pathnames, [int(0.7 * len(pathnames))])


def copy_images(data_dirname, class_dirname, image_pathnames):
  class_dirpath = os.path.join('data', data_dirname, class_dirname)

  if not os.path.exists(class_dirpath):
    os.makedirs(class_dirpath)

  for image_pathname in image_pathnames:
    image_filename = os.path.basename(image_pathname)
    shutil.copyfile(image_pathname,
                    os.path.join(class_dirpath, image_filename))


def split_images():
  for class_dirname in os.listdir('images'):
    image_dirpath = os.path.join('images', class_dirname)

    if not os.path.isdir(image_dirpath):
      continue

    train_pathnames, test_pathnames = split_pathnames(image_dirpath)

    copy_images('train', class_dirname, train_pathnames)
    copy_images('test', class_dirname, test_pathnames)

if __name__ == '__main__':
  clean_data()
  split_images()
