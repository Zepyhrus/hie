import cv2

from hie.tools import imread
from hie.hie import HIE
from hie.hieval import HIEval




if __name__ == "__main__":
  gt = HIE('data/hie/labels/hie.val.ann.json')
  dt = HIE('/home/ubuntu/Workspace/PoseBenchmark/det.json')

  hie_eval = HIEval(gt, dt, 'keypoints')

  hie_eval.evaluate()
  hie_eval.accumulate()
  hie_eval.summarize()

  msg, _ = hie_eval.new_summ()
  print(msg)

  hie_eval.viz(show_bbox=True)





