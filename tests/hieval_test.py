"""
This is the module test script. All modules need to be tested including:
* HIE.__init__(self, annotation_file, dataset);
* HIE._get_abs_name(self, image_id): included in self.img_files;
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

from hie.tools import imread, jsdump, jsload, GREEN
from hie.hie import HIE
from hie.hieval import HIEval


if __name__ == "__main__":
  # __init__
  gt = HIE('data/seed/labels/val128.json', 'seed')
  dt = gt.load_res('data/seed/labels/seed128.res.json')


  ev = HIEval(gt, dt, 'bbox')
  ev.new_summ()

  ev.viz(show_bbox=True)


  print('============== HIE module test passed! =============')

  





