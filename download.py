import os
import random
import re
import sys
import time
from urllib.request import urlretrieve


def download(channel):
  with open(f'urls/{channel}.txt', 'rt') as f:
    lines = f.readlines()

  dir = os.path.join('images', channel)
  if not os.path.exists(dir):
    os.makedirs(dir)

  for url in lines:
    name = re.findall(r'(?<=vi/).*(?=/hqdefault)', url)[0]
    path = os.path.join(dir, f'{name}.jpg')

    if os.path.exists(path):
      print(f'{path} already exists')
      continue

    print(f'download {path}')
    urlretrieve(url, path)
    time.sleep(1 + random.randint(0, 2))

if __name__ == '__main__':
  download(sys.argv[1])
