"""
This is the module test script. All modules need to be tested including:
* HIE.__init__(self, annotation_file, dataset);
* HIE._get_abs_name(self, image_id);
* HIE._strip(self);
* HIE._validate(self);
* HIE.createIndex(self): included in __init__();
* HIE.info(self);
* HIE.getAnnIds(self, imgIds, catIds, areaRng, iscrowd);
* HIE.getCatIds(self, catNms, supNms, catIds);
* HIE.getImgIds(self, imgIds, catIds);
* HIE.loadAnns(self, ids);
* HIE.loadCats(self, ids);
* HIE.loadImgs(self, ids);
* HIE.load_res(self, resFile);
* HIE.get_ann_by_name(self, name);
* HIE.viz_anns(self, img, anns, thresh, color, show_bbox, size): viz_anns() is included in HIE.viz()
* HIE.viz();
"""

import argparse

import cv2

from hie.tools import imread, jsdump, jsload
from hie.hie import HIE
from hie.hieval import HIEval


parser = argparse.ArgumentParser(description='test for hie module')
parser.add_argument('--ground-truth', '-g', dest='gt', type=str, default='data/hie/labels/hie.val.ann.json')
parser.add_argument('--detect', '-d', dest='dt', type=str, default='data/det.json')
parser.add_argument('--type', '-t', type=str, default='bbox')
parser.add_argument('--visualization', '-v', default=False, action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
  # __init__
  gt = HIE(args.gt)

  # _get_abs_name

  # _strip

  # _validate

  # info

  # getAnnIds

  # getCatIds

  # getImgIds

  # loadAnns

  # loadCats

  # loadImgs

  # load_res
  dt = gt.load_res('data/det.res.json')
  dt = gt.load_res(jsload('data/det.res.json'))

  





