import os
import csv
from config import Channel, LOG_DIR


def write_csv_file(dir):
  with open(os.path.join(dir, 'data.csv'), 'wt') as f:
    for i, channel in enumerate(Channel):
      image_dir = os.path.join(dir, channel.name)
      writer = csv.writer(f, lineterminator='\n')

      for filename in os.listdir(image_dir):
        writer.writerow([os.path.join(image_dir, filename), i])

if __name__ == '__main__':
  write_csv_file('data/train')
  write_csv_file('data/test')
