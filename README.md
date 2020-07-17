# hie
A repository inherited from COCO-api, for person detection/pose estimation benchmark.

This is an unofficial re-implementation of COCO-api, for the sake of solving the limitation of COCO-api including:
* The greedy matching seems to cause problem in matching process;
* Maximum detection is limited to 20 for keypoints/100 for detection validation;
* Parameters are casted implicitly in evaluation api;
* No visualization function;
* Other few tools maybe used in benchmark;

