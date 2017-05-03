import os
import csv
from config import CLASSES, LOG_DIR


def write_csv_file(dir):
  with open(os.path.join(dir, 'data.csv'), 'wt') as f:
    for i, classname in enumerate(CLASSES):
      image_dir = os.path.join(dir, classname)
      writer = csv.writer(f, lineterminator='\n')

      for filename in os.listdir(image_dir):
        writer.writerow([os.path.join(image_dir, filename), i])

if __name__ == '__main__':
  write_csv_file('data/train')
  write_csv_file('data/test')
