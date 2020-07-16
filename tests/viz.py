import cv2

from hie.tools import imread
from hie.hie import HIE
from hie.hieval import HIEval




if __name__ == "__main__":
  gt = HIE('data/hie/labels/hie.val.ann.json')
  dt = HIE('/home/ubuntu/Workspace/PoseBenchmark/backup/0520/0515_val_iou_0.5_thresh_0.4_float_.json')

  hie_eval = HIEval(gt, dt, 'bbox')

  hie_eval.evaluate()
  hie_eval.accumulate()
  hie_eval.summarize()

  # msg, _ = hie_eval.new_summ()
  # print(msg)





