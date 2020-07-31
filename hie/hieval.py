import datetime
import time
from collections import defaultdict
import copy
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment as KM

import cv2

from .tools import iou, imread, BLUE, GREEN, RED, YELLOW


class HIEParams(object):
  '''
  Params for coco evaluation api
  '''
  def setDetParams(self):
    self.imgIds = []
    self.catIds = []
    # np.arange causes trouble.  the data point on arange is slightly larger than the true value
    self.iouThrs = np.arange(50, 100, 5)  / 100
    self.recThrs = np.arange( 0, 101)     / 100
    self.maxDets = [1, 10, 100]
    self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    self.areaRngLbl = ['all', 'small', 'medium', 'large']
    self.useCats = 1

  def setKpParams(self):
    self.imgIds = []
    self.catIds = []
    # np.arange causes trouble.  the data point on arange is slightly larger than the true value
    self.iouThrs = np.arange(50, 100, 5)  / 100
    self.recThrs = np.arange( 0, 101)     / 100
    self.maxDets = [256]
    self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    self.areaRngLbl = ['all', 'medium', 'large']
    self.useCats = 1

  def __init__(self, iouType='segm'):
    if iouType == 'segm' or iouType == 'bbox':
      self.setDetParams()
    elif iouType == 'keypoints':
      self.setKpParams()
    else:
      raise Exception('iouType not supported')
    self.iouType = iouType
    # useSegm is deprecated
    self.useSegm = None

    if iouType == 'keypoints':
      self.kpt_oks_sigmas = np.array([79, 79, 71, 72, 62, 79, 72, 62, 107, 87, 89, 107, 87, 89]) * 1.0 / 1000

  def info(self):
    for key, value in vars(self).items():
      print(f'{key}: {value}')


class HIEval(object):
  """
  """
  def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints'):
    '''
    Initialize CocoEval using coco APIs for gt and dt
    :param cocoGt: coco object with ground truth annotations
    :param cocoDt: coco object with detection results
    :return: None
    '''
    # TODO: add evaluation for other type of iouType
    assert iouType in ['bbox', 'keypoints'], 'Other type of evaluation is not supported yet!'

    self.cocoGt   = cocoGt                    # ground truth COCO API
    self.cocoDt   = cocoDt                    # detections COCO API
    self.params   = {}                        # evaluation parameters
    self.evalImgs = defaultdict(list)         # per-image per-category evaluation results [KxAxI] elements
    self.eval     = {}                        # accumulated evaluation results
    self._gts = defaultdict(list)             # gt for evaluation
    self._dts = defaultdict(list)             # dt for evaluation
    self.params = HIEParams(iouType=iouType)  # parameters
    self._paramsEval = {}                     # parameters for evaluation
    self.stats = []                           # result summarization
    self.ious = {}                            # ious between all gts and dts

    
    self.iou_type = iouType
    self.MATCH = defaultdict(list)
    self.__evaluated = False

    if not cocoGt is None:
      self.params.imgIds = sorted(cocoGt.getImgIds())
      self.params.catIds = sorted(cocoGt.getCatIds())

  def _prepare(self):
    '''
    Prepare ._gts and ._dts for evaluation based on params
    :return: None
    '''
    def _toMask(anns, coco):
        # modify ann['segmentation'] by reference
        for ann in anns:
            rle = coco.annToRLE(ann)
            ann['segmentation'] = rle
    p = self.params
    if p.useCats:
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
    else:
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

    # convert ground truth to mask if iouType == 'segm'
    if p.iouType == 'segm':
        _toMask(gts, self.cocoGt)
        _toMask(dts, self.cocoDt)
    # set ignore flag
    for gt in gts:
        gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
        gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        if p.iouType == 'keypoints':
            gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
    self._gts = defaultdict(list)       # gt for evaluation
    self._dts = defaultdict(list)       # dt for evaluation
    for gt in gts:
        self._gts[gt['image_id'], gt['category_id']].append(gt)
    for dt in dts:
        self._dts[dt['image_id'], dt['category_id']].append(dt)
    self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
    self.eval     = {}                  # accumulated evaluation results

  def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    tic = time.time()
    print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
      p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
      print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
      p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params=p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    self.ious = {
      (imgId, catId): self.computeOKS(imgId, catId) \
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    self.evalImgs = [
      evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc-tic))

  def computeOKS(self, imgId, catId):
    p = self.params
    if p.useCats:
      gt = self._gts[imgId,catId]
      dt = self._dts[imgId,catId]
    else:
      gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
      dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    if len(gt) == 0 and len(dt) ==0:
      return []
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
      dt=dt[0:p.maxDets[-1]]
    
    # compute iou between each dt and gt region
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = np.zeros((len(dt), len(gt)))

    for i, d in enumerate(dt):
      for j, g in enumerate(gt):
        ious[i, j] = self.oks(g, d, method=p.iouType)

    return ious

  def evaluateImg(self, imgId, catId, aRng, maxDet):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    p = self.params
    if p.useCats:
      gt = self._gts[imgId,catId]
      dt = self._dts[imgId,catId]
    else:
      gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
      dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    if len(gt) == 0 and len(dt) ==0:
      return None

    for g in gt:
      if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
        g['_ignore'] = 1
      else:
        g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm  = np.zeros((T,G))
    dtm  = np.zeros((T,D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T,D))
    if not len(ious)==0:
      for tind, t in enumerate(p.iouThrs):
        for dind, d in enumerate(dt):
          # information about best match so far (m=-1 -> unmatched)
          iou = min([t,1-1e-10])
          m   = -1
          for gind, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if gtm[tind,gind]>0 and not iscrowd[gind]:
              continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
              break
            # continue to next gt unless better match made
            if ious[dind,gind] < iou:
              continue
            # if match successful and best so far, store appropriately
            iou=ious[dind,gind]
            m=gind
          # if match made store id of match for both dt and gt
          if m ==-1:
            continue
          dtIg[tind,dind] = gtIg[m]
          dtm[tind,dind]  = gt[m]['id']
          gtm[tind,m]     = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
    # store results for given image and category
    return {
      'image_id':     imgId,
      'category_id':  catId,
      'aRng':         aRng,
      'maxDet':       maxDet,
      'dtIds':        [d['id'] for d in dt],
      'gtIds':        [g['id'] for g in gt],
      'dtMatches':    dtm,
      'gtMatches':    gtm,
      'dtScores':     [d['score'] for d in dt],
      'gtIgnore':     gtIg,
      'dtIgnore':     dtIg,
    }

  def accumulate(self, p = None):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
      print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
      p = self.params

    p.catIds = p.catIds if p.useCats == 1 else [-1]
    T           = len(p.iouThrs)
    R           = len(p.recThrs)
    K           = len(p.catIds) if p.useCats else 1
    A           = len(p.areaRng)
    M           = len(p.maxDets)
    precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    recall      = -np.ones((T,K,A,M))
    scores      = -np.ones((T,R,K,A,M))

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
      Nk = k0*A0*I0
      for a, a0 in enumerate(a_list):
        Na = a0*I0
        for m, maxDet in enumerate(m_list):
          E = [self.evalImgs[Nk + Na + i] for i in i_list]
          E = [e for e in E if not e is None]
          if len(E) == 0:
            continue
          dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

          # different sorting method generates slightly different results.
          # mergesort is used to be consistent as Matlab implementation.
          inds = np.argsort(-dtScores, kind='mergesort')
          dtScoresSorted = dtScores[inds]

          dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
          dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
          gtIg = np.concatenate([e['gtIgnore'] for e in E])
          npig = np.count_nonzero(gtIg==0 )
          if npig == 0:
            continue
          tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
          fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

          tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
          fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
          for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp = np.array(tp)
            fp = np.array(fp)
            nd = len(tp)
            rc = tp / npig
            pr = tp / (fp+tp+np.spacing(1))
            q  = np.zeros((R,))
            ss = np.zeros((R,))

            if nd:
              recall[t,k,a,m] = rc[-1]
            else:
              recall[t,k,a,m] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            pr = pr.tolist(); q = q.tolist()

            for i in range(nd-1, 0, -1):
              if pr[i] > pr[i-1]:
                pr[i-1] = pr[i]

            inds = np.searchsorted(rc, p.recThrs, side='left')
            try:
              for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
                ss[ri] = dtScoresSorted[pi]
            except:
              pass
            precision[t,:,k,a,m] = np.array(q)
            scores[t,:,k,a,m] = np.array(ss)
    self.eval = {
      'params': p,
      'counts': [T, R, K, A, M],
      'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      'precision': precision,
      'recall':   recall,
      'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format( toc-tic))

  def summarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
      p = self.params
      iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
      titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
      typeStr = '(AP)' if ap==1 else '(AR)'
      iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
          if iouThr is None else '{:0.2f}'.format(iouThr)

      aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
      mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
      if ap == 1:
          # dimension of precision: [TxRxKxAxM]
          s = self.eval['precision']
          # IoU
          if iouThr is not None:
              t = np.where(iouThr == p.iouThrs)[0]
              s = s[t]
          s = s[:,:,:,aind,mind]
      else:
          # dimension of recall: [TxKxAxM]
          s = self.eval['recall']
          if iouThr is not None:
              t = np.where(iouThr == p.iouThrs)[0]
              s = s[t]
          s = s[:,:,aind,mind]
      if len(s[s>-1])==0:
          mean_s = -1
      else:
          mean_s = np.mean(s[s>-1])
      print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
      return mean_s
    
    def _summarizeDets():
      stats = np.zeros((12,))
      stats[0] = _summarize(1)
      stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
      stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
      stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
      stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
      stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
      stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
      stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
      stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
      stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
      stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
      stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
      return stats
    def _summarizeKps():
      stats = np.zeros((10,))
      stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
      stats[1] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.5)
      stats[2] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.75)
      stats[3] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='medium')
      stats[4] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='large')
      stats[5] = _summarize(0, maxDets=self.params.maxDets[0])
      stats[6] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.5)
      stats[7] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.75)
      stats[8] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='medium')
      stats[9] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='large')
      return stats

    if not self.eval:
      raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
      summarize = _summarizeDets
    elif iouType == 'keypoints':
      summarize = _summarizeKps

    self.stats = summarize()

  def new_eval(self):
    """
    Custom evaluation using KM algorithm, generating potential matches
    """
    # load_res ensures images in dt is subset of gt
    gt = self.cocoGt
    dt = self.cocoDt
    img_ids = gt.getImgIds()

    print('----------------- start evaluating -------------------------')
    for img_id in tqdm(img_ids):
      gts = gt.loadAnns(gt.getAnnIds([img_id]))
      dts = dt.loadAnns(dt.getAnnIds([img_id]))

      if len(gts) == 0 or len(dts) == 0:
        self.MATCH[img_id] = []
        continue

      self.MATCH[img_id] = self.metric_by_frame(gts, dts, self.iou_type)
    print('--------------- evaluation finished! -----------------------')
    self.__evaluated = True

  def __check_evaluated(self):
    if not self.__evaluated:
      print('[WRN] current dataset not evaluated, start evaluation.')
      self.new_eval()

  def new_summ(self, img_ids=[]):
    """
    Summarize using new evaluation method
    """
    self.__check_evaluated()
    
    if len(img_ids) == 0:
      img_ids = self.cocoGt.getImgIds()
    
    # initialize summary matrix
    _summary = {}
    for i in range(10):
      # with respect to iou from 0.5 to 0.95
      _summary[str(50 + 5 * i)] = {
        'TP': 0,  # true Positive
        'FP': 0,  # mAP
        'FN': 0,  # mAR
      }
    _summary['avg'] = {'TP': 0, 'FP': 0, 'FN': 0}

    for img_id in img_ids:
      _match = self.MATCH[img_id]
      num_gts = len(self.cocoGt.getAnnIds([img_id]))
      num_dts = len(self.cocoDt.getAnnIds([img_id]))

      for i in range(10):
        _thresh = 0.5 + 0.05 * i

        _tp = len([_ for _ in _match if _['oks'] >= _thresh])
        _fp = num_dts - _tp
        _fn = num_gts - _tp

        assert _tp >= 0 and _fp >= 0 and _fn >= 0, 'Debug here!'

        _summary[str(50 + 5 * i)]['TP'] += _tp
        _summary[str(50 + 5 * i)]['FP'] += _fp
        _summary[str(50 + 5 * i)]['FN'] += _fn

        _summary['avg']['TP'] += _tp
        _summary['avg']['FP'] += _fp
        _summary['avg']['FN'] += _fn
    
    _mAP    = _summary['avg']['TP'] / ( _summary['avg']['FP'] + _summary['avg']['TP'] + 1e-6) * 100
    _mAP_5  = _summary['50']['TP']  / ( _summary['50']['FP']  + _summary['50']['TP']  + 1e-6) * 100
    _mAP_75 = _summary['75']['TP']  / ( _summary['75']['FP']  + _summary['75']['TP']  + 1e-6) * 100
    _mAP_90 = _summary['90']['TP']  / ( _summary['90']['FP']  + _summary['90']['TP']  + 1e-6) * 100
    _mAR    = _summary['avg']['TP'] / ( _summary['avg']['FN'] + _summary['avg']['TP'] + 1e-6) * 100
    _mAR_5  = _summary['50']['TP']  / ( _summary['50']['FN']  + _summary['50']['TP']  + 1e-6) * 100
    _mAR_75 = _summary['75']['TP']  / ( _summary['75']['FN']  + _summary['75']['TP']  + 1e-6) * 100
    _mAR_90 = _summary['90']['TP']  / ( _summary['90']['FN']  + _summary['90']['TP']  + 1e-6) * 100

    _mf1 = 2 * _mAP * _mAR / (_mAP + _mAR + 1e-6)

    msg = f'mAP: {_mAP:.2f} mAR: {_mAR:.2f} mF1: {_mf1:.2f}'

    return msg, _summary

  def new_combine(self, prob=0.1):
    self.__check_evaluated()

    img_ids = sorted(self.cocoGt.getImgIds())
    combined_anns = []

    # smooth correct detection
    for img_id in tqdm(img_ids):
      _match = self.MATCH[img_id]

      _matched_gt     = [_['gt_id'] for _ in _match]
      _matched_dt     = [_['dt_id'] for _ in _match]
      _matched_score  = [_['oks']   for _ in _match]

      for _gi, _di in zip(_matched_gt, _matched_dt):
        gt = self.cocoGt.loadAnns([_gi])[0]
        dt = self.cocoDt.loadAnns([_di])[0]

        combined_ann = copy.deepcopy(gt)
        combined_ann['bbox'] = [(g+d)/2 for g, d in zip(gt['bbox'], dt['bbox'])]

        combined_anns.append(combined_ann)
    
      # add false negtives
      for gt in self.cocoGt.loadAnns(self.cocoGt.getAnnIds([img_id])):
        if gt['id'] in _matched_gt:
          continue

        combined_anns.append(gt)

      # add false positives
      for dt in self.cocoDt.loadAnns(self.cocoDt.getAnnIds([img_id])):
        if dt['id'] in _matched_dt:
          continue

        if np.random.rand() <= prob:
          combined_anns.append(dt)

    return self.cocoGt.load_res(combined_anns)
    
    
  def viz(self, img_ids=[], show_bbox=False, show_bbox_only=True, shuffle=False, thresh=0.75, resize=True):
    self.__check_evaluated()

    if self.iou_type == 'bbox' and not show_bbox:
      print('[WRN]: IoU type bbox but not showing bbox!')

    if show_bbox_only and not show_bbox:
      print('[WRN]: Not showing anything!')

    if len(img_ids) == 0:
      img_ids = self.cocoGt.getImgIds()

    if shuffle:
      random.shuffle(img_ids)
    else:
      img_ids = sorted(img_ids)

    for img_id in img_ids:
      img = imread(self.cocoGt._get_abs_name(img_id))

      _match = self.MATCH[img_id]

      _matched_gt     = [_['gt_id'] for _ in _match]
      _matched_dt     = [_['dt_id'] for _ in _match]
      _matched_score  = [_['oks']   for _ in _match]

      for _gi, _di, _s in zip(_matched_gt, _matched_dt, _matched_score):
        gt = self.cocoGt.loadAnns([_gi])[0]
        dt = self.cocoDt.loadAnns([_di])[0]

        if _s >= thresh:
          img = self.cocoGt.viz_anns(img, [gt], color=BLUE, show_bbox=show_bbox, show_bbox_only=show_bbox_only)
          img = self.cocoDt.viz_anns(img, [dt], color=YELLOW, show_bbox=show_bbox, show_bbox_only=show_bbox_only)
        else:
          img = self.cocoGt.viz_anns(img, [gt], color=GREEN, show_bbox=show_bbox, show_bbox_only=show_bbox_only)
          img = self.cocoDt.viz_anns(img, [dt], color=RED, show_bbox=show_bbox, show_bbox_only=show_bbox_only)

      for gt in self.cocoGt.loadAnns(self.cocoGt.getAnnIds([img_id])):
        if gt['id'] in _matched_gt:
          continue

        img = self.cocoGt.viz_anns(img, [gt], color=GREEN, show_bbox=show_bbox, show_bbox_only=show_bbox_only)
      
      for dt in self.cocoDt.loadAnns(self.cocoDt.getAnnIds([img_id])):
        if dt['id'] in _matched_dt:
          continue

        img = self.cocoDt.viz_anns(img, [dt], color=RED, show_bbox=show_bbox, show_bbox_only=show_bbox_only)

      if resize:
        scale = np.sqrt(img.shape[0] * img.shape[1] / 960 / 810)
        img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))

      print('Image id: ' + img_id)
      cv2.imshow('_', img)
      if cv2.waitKey(0) == 27: break
  
  def oks(self, gt, dt, method='keypoints'):
    '''
    calculate IOU/OKS between specific ground truth and detection
    '''
    if method == 'bbox':
      return iou(gt['bbox'], dt['bbox'], xywh=True)
    elif method == 'keypoints':
      var = (self.params.kpt_oks_sigmas * 2) ** 2

      g = np.array(gt['keypoints'])
      xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
      k1 = np.count_nonzero(vg > 0)

      x, y, w, h = gt['bbox']
      x0 = x-w; x1 = x+w*2
      y0 = y-h; y1 = y+w*2

      d = np.array(dt['keypoints'])
      xd = d[0::3]; yd = d[1::3]
      if k1 > 0:
        dx = xd - xg; dy = yd - yg
      else:
        z = np.zeros((len(self.params.kpt_oks_sigmas)))
        dx = np.max((z, x0-xd), axis=0)+np.max((z, xd-x1), axis=0)
        dy = np.max((z, y0-yd), axis=0)+np.max((z, yd-y1), axis=0)

      e = (dx**2 + dy**2) / var / (gt['area'] + np.spacing(1)) / 2
      
      if k1 > 0:
        e = e[vg > 0]

      _oks = np.mean([np.exp(- i) if i <= 10 else 0 for i in e])

      return _oks
    else:
      raise NotImplementedError("IOU method not implemented!")

  def metric_by_frame(self, gts, dts, method='keypoints'):
    """
      return matched ground truth and detects with only scores exceeding 0.5 OKS
    """
    assert set([_['image_id'] for _ in gts]) == set([_['image_id'] for _ in dts]), 'gt and dt from different image!'
    assert len(gts), "No ground truth!"
    assert len(dts), "No detections!"

    metric = - np.ones((len(gts), len(dts)))

    for i in range(len(gts)):
      for j in range(len(dts)):
        metric[i, j] = self.oks(gts[i], dts[j], method)


    gt_idx, dt_idx = KM(metric, maximize=True)
    match = []

    for i, j in zip(gt_idx, dt_idx):
      if metric[i, j] >= 0.5:
        match.append({
          'gt_id': gts[i]['id'],
          'dt_id': dts[j]['id'],
          'oks':  metric[i, j]
        })

    return match

  def __str__(self):
    self.summarize()