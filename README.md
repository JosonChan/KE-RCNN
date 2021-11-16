# KE-R-CNN

Official implementation of **KE-R-CNN** for part-level attribute parsing.

## Installation
- pytorch 1.8.1
- python 3.7.0
- mmdetection 2.17.0
- [fashionpeida-API](https://github.com/KMnP/fashionpedia-api)

## Dataset
You need to download the datasets and annotations follwing this repo's formate

- [FashionPedia](https://github.com/cvdfoundation/fashionpedia)

Make sure to put the files as the following structure:

```
  ├─data
  │  ├─fashionpedia
  │  │  ├─train
  │  │  ├─test
  │  │  │─instances_attribute_train2020.json
  │  │  │─instances_attribute_val2020.json
  |
  ├─work_dirs
  |  ├─KE-RCNN_r50_1x
  |  |  ├─latest.pth
  ```

## Results and Models

|  Backbone    |  LR  | AP_iou/AP_iou+f1 | AP_mask_iou/AP_mask_iou+f1 | DOWNLOAD |
|--------------|:----:|:----------------:|:--------------------------:|:--------:|
|  R-50        |  1x  | 41.9/39.1        | 37.5/36,2                  |          |
|  R-101       |  1x  | 43.8/39.9        | 38.2/36.0                  |          |
|  Cascade-R-50|  1x  | 44.0/41.0        | 37.5/36.5                  |          |
|  Swim-tiny   |  1x  | 45.0/41.5        | 40.6/38.6                  |          |

## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/relation_rcnn/relation_rcnn_r50_1x_fashion_v4.py ./work_dirs/relation_rcnn_r50_1x_fashion_v4/epoch_32.pth 8 --format-only --eval-options "jsonfile_prefix=./KE-RCNN_r50_1x_val_result"

# eval, noted that should change the json path produce by previous step.
python eval/fashion_eval.py
```

## Training

Coming soon...