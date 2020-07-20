"""
This is the module test script. All modules need to be tested including:
* 
"""





import argparse

import cv2

from hie.tools import imread
from hie.hie import HIE
from hie.hieval import HIEval


parser = argparse.ArgumentParser(description='test for hie module')
parser.add_argument('--ground-truth', '-g', dest='gt', type=str, default='data/hie/labels/hie.val.ann.json')
parser.add_argument('--detect-result', '-d', dest='dt', type=str, default='/home/ubuntu/Workspace/PoseBenchmark/det.json')
parser.add_argument('--type', '-t', type=str, default='bbox')
parser.add_argument('--visualization', '-v', default=False, action='store_true')
args = parser.parse_args()



if __name__ == "__main__":
  gt = HIE(args.gt)
  dt = HIE(args.dt)

  num_instances = [len(dt.getAnnIds([_])) for _ in dt.getImgIds()]

  print(f'Maximum number of instances in detection is: {max(num_instances)}.')

  hie_eval = HIEval(gt, dt, args.type)

  hie_eval.evaluate()
  hie_eval.accumulate()
  hie_eval.summarize()

  msg, _ = hie_eval.new_summ()
  print(msg)

  if args.visualization:
    hie_eval.viz(show_bbox=True)





