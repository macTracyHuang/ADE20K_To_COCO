# ADE20K_To_COCO
This tool can convert neweast [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) to COCO format, which can be futher used in Detectron2 Framework.
As my goal is to build a object level SLAM system, this tool is designed to retrieve information realted to target objects.
However, you can always modify it to fit your purposes, and Welcome to pull a request.

## Variables
objectNames: Name of target objects, ex: ["door", "door frame"]\
pklPath: the path to index_ade20k.pkl provied by ADE20K\
datasetDir: ADE20K dataset path

## Test Result
Use demo.ipynb to visualize if it works\
![image](./figure/door1.png)\
![image](./figure/wall1.png)\