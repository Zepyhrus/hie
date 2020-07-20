__author__ = 'sherk'
__version__ = '0.1.2'
"""

"""

import json
import os
import sys
from collections import defaultdict
import itertools
import random
import time
import copy

import cv2

from .tools import BLUE, GREEN, RED, YELLOW, PURPLE, BROWN, COLORS, PAIRS, imread

def _isArrayLike(obj):
  return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class HIE(object):
  def __init__(self, annotation_file=None, dataset='hie'):
    """
    Constructor of Microsoft COCO helper class for reading and visualizing annotations.
    :param annotation_file (str): location of annotation file
    :param image_folder (str): location to the folder that hosts images.
    :return:
    """
    # load dataset
    self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
    self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
    self.orm = dataset
    self.names = None

    if not annotation_file == None:
      # print('loading annotations into memory...')
      dataset = json.load(open(annotation_file, 'r'))
      assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))

      self.dataset = dataset
      self.createIndex()
    
  def _get_abs_name(self, image_id):
    if self.orm == 'hie':
      return f'data/hie/images/train/{image_id}.jpg'
    elif self.orm == 'cpose':
      return f'data/cpose/images/{image_id}.jpg'
    elif self.orm == 'coco':
      return f'data/coco/train2017/{image_id:012d}.jpg'
    elif self.orm == 'hieval':
      return f'data/hie/images/val/{image_id}.jpg'
    else:
      raise NotImplementedError('Current dataset is not implemented!')

  def _strip(self):
    """
    strip images without annotation
    """
    _img_ids = set([_['image_id'] for _ in self.dataset['annotations']])

    self.dataset['images'] = [_ for _ in self.dataset['images'] if _['id'] in _img_ids]
    self.createIndex()
  
  def _validate(self):
    """
    """
    for ann in self.dataset['annotations']:
      x, y, w, h = ann['bbox']
      xs = ann['keypoints'][::3]
      ys = ann['keypoints'][1::3]
      vs = ann['keypoints'][2::3]

      for i, (_x, _y, _v) in enumerate(zip(xs, ys, vs)):
        if _v < 0.05: continue

        if not ((x-w*0.1 < _x < x+w*1.1) and (y-h*0.1 < _y < y+h*1.1)):
          print('[WRN]: ', ann['image_id'], ' @ ', i)
  
  def createIndex(self):
    # create index
    anns, cats, imgs = {}, {}, {}
    imgToAnns,catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
      for ann in self.dataset['annotations']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann

    if 'images' in self.dataset:
      for img in self.dataset['images']:
        imgs[img['id']] = img

    if 'categories' in self.dataset:
      for cat in self.dataset['categories']:
        cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
      for ann in self.dataset['annotations']:
        catToImgs[ann['category_id']].append(ann['image_id'])

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats
  
  def info(self):
    """
    Print information about the annotation file.
    :return:
    """
    if 'info' in self.dataset:
      for key, value in self.dataset['info'].items():
        print('{}: {}'.format(key, value))
    else:
      print('No available info message!')

  def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
    """
    Get ann ids that satisfy given filter conditions. default skips that filter
    :param:
      imgIds  (int array)     : get anns for given imgs
      catIds  (int array)     : get anns for given cats
      areaRng (float array)   : get anns for given area range (e.g. [0 inf])
      iscrowd (boolean)       : get anns for given crowd label (False or True)
    :return: ids (int array)       : integer array of ann ids
    """
    imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
    catIds = catIds if _isArrayLike(catIds) else [catIds]

    if len(imgIds) == len(catIds) == len(areaRng) == 0:
      anns = self.dataset['annotations']
    else:
      if not len(imgIds) == 0:
        lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))
      else:
        anns = self.dataset['annotations']
      anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
      anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
    if not iscrowd == None:
      ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
    else:
      ids = [ann['id'] for ann in anns]
    return ids

  def getCatIds(self, catNms=[], supNms=[], catIds=[]):
    """
    filtering parameters. default skips that filter.
    :param catNms (str array)  : get cats for given cat names
    :param supNms (str array)  : get cats for given supercategory names
    :param catIds (int array)  : get cats for given cat ids
    :return: ids (int array)   : integer array of cat ids
    """
    catNms = catNms if _isArrayLike(catNms) else [catNms]
    supNms = supNms if _isArrayLike(supNms) else [supNms]
    catIds = catIds if _isArrayLike(catIds) else [catIds]

    if len(catNms) == len(supNms) == len(catIds) == 0:
      cats = self.dataset['categories']
    else:
      cats = self.dataset['categories']
      cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
      cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
      cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
    ids = [cat['id'] for cat in cats]
    return ids

  def getImgIds(self, imgIds=[], catIds=[]):
    '''
    Get img ids that satisfy given filter conditions.
    :param imgIds (int array) : get imgs for given ids
    :param catIds (int array) : get imgs with all given cats
    :return: ids (int array)  : integer array of img ids
    '''
    imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
    catIds = catIds if _isArrayLike(catIds) else [catIds]

    if len(imgIds) == len(catIds) == 0:
      ids = self.imgs.keys()
    else:
      ids = set(imgIds)
      for i, catId in enumerate(catIds):
        if i == 0 and len(ids) == 0:
          ids = set(self.catToImgs[catId])
        else:
          ids &= set(self.catToImgs[catId])
    return list(ids)

  def loadAnns(self, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying anns
    :return: anns (object array) : loaded ann objects
    """
    if _isArrayLike(ids):
      return [self.anns[id] for id in ids]
    elif type(ids) == int:
      return [self.anns[ids]]

  def loadCats(self, ids=[]):
    """
    Load cats with the specified ids.
    :param ids (int array)       : integer ids specifying cats
    :return: cats (object array) : loaded cat objects
    """
    if _isArrayLike(ids):
      return [self.cats[id] for id in ids]
    elif type(ids) == int:
      return [self.cats[ids]]

  def loadImgs(self, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying img
    :return: imgs (object array) : loaded img objects
    """
    if _isArrayLike(ids):
      return [self.imgs[id] for id in ids]
    elif type(ids) == int:
      return [self.imgs[ids]]

  def load_res(self, resFile):
    """
    Load result file and return a HIE result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = HIE()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    tic = time.time()
    if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
      anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
      anns = self.loadNumpyAnnotations(resFile)
    else:
      anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
      'Results do not correspond to current hie set'

    if 'caption' in anns[0]:
      imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
      res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
      for id, ann in enumerate(anns):
        ann['id'] = id+1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for i, ann in enumerate(anns):
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if not 'segmentation' in ann:
          ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        ann['area'] = bb[2]*bb[3]
        ann['id'] = i+1
        ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for id, ann in enumerate(anns):
        # now only support compressed RLE format as segmentation results
        ann['area'] = maskUtils.area(ann['segmentation'])
        if not 'bbox' in ann:
          ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
        ann['id'] = id+1
        ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for i, ann in enumerate(anns):
        s = ann['keypoints']
        x = s[0::3]
        y = s[1::3]
        x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
        ann['area'] = (x1-x0)*(y1-y0)
        ann['id'] = i + 1
        ann['bbox'] = [x0,y0,x1-x0,y1-y0]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    res.createIndex()
    return res

  def get_ann_by_name(self, name=None):
    """ TODO: get annotations by SINGLE name and sort them by image"""
    assert name is not None, 'You must provide a name to get annotations!'
    
    print('This function is not implemented yet!')

  def viz_anns(self, img, anns=[], thresh=0.05, color=None, show_bbox=False, size=1):
    if 'skeleton' in self.cats[1]:
      limbs = self.cats[1]['skeleton']
    else:
      limbs = PAIRS
    new_img = img.copy()

    for ann in anns:
      if color is None:
        _color = tuple(int(_) for _ in random.choice(COLORS))
      else:
        _color = color
      
      if 'bbox' in ann and show_bbox:
        x, y, w, h = [int(_) for _ in ann['bbox']]

        cv2.rectangle(new_img, (x, y), (x+w, y+h), _color, 1)

      if 'keypoints' in ann:
        xs = [int(_) for _ in ann['keypoints'][0::3]]
        ys = [int(_) for _ in ann['keypoints'][1::3]]
        vs = ann['keypoints'][2::3]

        for i, (x, y, v) in enumerate(zip(xs, ys, vs)):
          # if i == 0 and 'name' in ann:
          #   name = ann['name'].split('-')[-1]
          #   cv2.putText(new_img, name, (x, y), 0, 0.5, _color, 1)

          # if i == 1 and 'score' in ann:
          #   cv2.putText(new_img, '%.2f'%ann['score'], (x, y), 0, 0.5, _color, 1)
          
          if v > thresh:
            if i in [2, 3, 4, 8, 9, 10]:
              cv2.circle(new_img, (x, y), 0, _color, 6)
            else:
              cv2.circle(new_img, (x, y), 0, _color, 2)
            # cv2.putText(new_img, str('%d:%.2f' % (i, v)), (x, y), 0, 0.5, _color, 1)
            
          
        for limb in limbs:
          if vs[limb[0]] > thresh and vs[limb[1]] > thresh:
            st = (xs[limb[0]], ys[limb[0]])
            ed = (xs[limb[1]], ys[limb[1]])
            
            cv2.line(new_img, st, ed, _color, size)

    return new_img
  
  def viz(self, imgIds=[], thresh=0.05, dataset='hie', show_bbox=False, color=None, pause=0, shuffle=False):
    if not imgIds:
      imgIds = self.getImgIds()
    
    if not shuffle:
      imgIds = sorted(imgIds)
    else:
      random.shuffle(imgIds)
    
    for img_id in imgIds:
      print('image_id', img_id)
      annIds = self.getAnnIds(imgIds=[img_id]) # a 7 people image

      anns = self.loadAnns(annIds)

      if dataset == 'crowdpose':
        img = imread(self._get_abs_name(img_id))
      elif dataset == 'hie':
        img = imread(self._get_abs_name(img_id))
      else:
        raise NotImplementedError(f'current dataset {dataset} is not implemented!')

      img = self.viz_anns(img, anns=anns, thresh=thresh, show_bbox=show_bbox, color=color, size=2)

      if img.shape[0] * img.shape[1] < 640 * 480:
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
      elif img.shape[0] * img.shape[1] > 1440 * 1080:
        img = cv2.resize(img, (img.shape[1]*2//3, img.shape[0]*2//3))
      
      cv2.imshow('_', img)
      if cv2.waitKey(pause) == 27: break


if __name__ == "__main__":
  pass