# hie
A repository inherited from COCO-api, for person detection/pose estimation benchmark.

## Intro

This is an unofficial re-implementation of COCO-api.

**Why do we need another version of COCO-api?**: For the sake of solving the limitation of COCO-api including:
* The greedy-search matching, used in COCO api seems to cause problem in matching process;
* Maximum detection is limited to 20 for keypoints/100 for detection validation;
* Parameters are casted implicitly in evaluation api, very hard to re-implement to adapt to different datasets;
* No visualization function;
* Add few tools maybe used in benchmark;



## How-to-use-it

**Main functions**: Basically, **hie** project inherits from COCO-api, it contains every function included in COCO and also the original evaluation method in COCO.

**tools**: Some basic simple functions often used in objection detction/pose estimation benchmark.





## Versions

**0.0.0**: The original version was derived from PoseBenchmark project;

**0.1.0**: Initialization;

**0.1.1**: Replace original COCO-api: ComputeIOU with HIE-api: oks;

**0.1.2**: Add test functions;

**0.1.3**: Fix `time` and `copy` bug;

**0.1.4**: Fix `PYTHONVERSION` bug;

**0.1.5**: Add `HIE.load_res()` test;


## TODOs

1. Add function return absolute paths of all images;
2. Add `mot` support;
  * `_get_abs_name` support for mot;
  * remove `hieval` for dataset format;