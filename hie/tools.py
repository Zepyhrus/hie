
import numpy as np

import cv2
from turbojpeg import TurboJPEG


BLUE    = ( 255,   0,   0)
GREEN   = (   0, 255,   0)
RED     = (   0,   0, 255)
YELLOW  = (   0, 255, 255)
PURPLE  = ( 255,   0, 255)
BROWN   = ( 255, 255, 255)

COLORS = tuple(map(lambda x: tuple(int(_) for _ in x), np.random.randint(255, size=(32, 3))))

PAIRS = [
  [  0,   1 ], [  1,   2 ], [  3,   4 ], [  4,   5 ],
  [  6,   7 ], [  7,   8 ], [  9,  10 ], [ 10,  11 ],
  [ 12,  13 ], [  0,  13 ], [ 13,   3 ], [  3,   9 ],
  [  9,   6 ], [  6,   0 ]
]

def imread(image):
  try:
    with open(image, 'rb') as _i:
      return TurboJPEG.decode(_i.read())
  except:
    return cv2.imread(image)


def iou(box1, box2, xywh=False):
  '''
  if xywh:
    box1, box2 in x, y, w, h format
  else:
    box1, box2 in x1, y1, x2, y2 format
  '''
  if xywh:
    x1, y1, w1, h1 = box1
    a1, b1, c1, d1 = box2

    x2 = x1 + w1
    y2 = y1 + h1

    a2 = a1 + c1
    b2 = b1 + d1
  else:
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

  inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
  _i = inter / ((x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter + 1e-6)

  return _i