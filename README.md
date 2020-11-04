# SimplePose
该项目主要包含一系列Top-Down的2D姿态估计算法,如[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) 中的DUC,[Simple Baselines](https://arxiv.org/abs/1804.06208) 中DCONV,
以及[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) HRNet.同时也包含了[DarkPose](https://github.com/ilovepose/DarkPose) 所提到的KeyPoints Encoder与Decoder的
一些小的tricks(大约获得1.0左右的mAP增益).


## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
## result
该项目以192x256的图片(包含人体的图片patch)作为输入,使用了4块显卡进行训练,batch_size=128(32/卡,显存17892MB,约17GB).总epoch=180,
初始学习率为0.001,使用了Adam作为优化器.学习率在第120个epoch与第160个epoch进行衰减,衰减系数0.1,训练时长约为21h(小时).测试所使用的heat_map到
keypoints的decode为具体解码器为 GaussTaylorKeyPointDecoder

**以下结果均使用[COCO_val2017_detections_AP_H_56_person.json](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) 的检测结果作为测试,数据来自
[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)**

### DConv(上采样使用转置卷积)的performance
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.714
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.913
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.789
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.688
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.760
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.752
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.815
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.719
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.802
```
### DUC(上采样Conv+PixelShuffle)的performance)
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.726
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.923
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.801
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.698
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.930
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.823
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.724
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.813
```
### SELayer+DUC(上采样Conv+PixelShuffle)的performance
**模型初始化时的reduction为True**
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.734
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.923
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.810
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.706
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.777
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.768
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.932
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.831
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.734
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.818
```


## training
目前支持coco 关键点数据集.自定义数据集请参考datasets/coco.py中的MSCOCO.__load_in()部分代码.相信这部分代码非常容易改写.

### COCO
* modify main.py (modify config file path)
```python
from processors.dp_pose_resnet_solver import DPProcessor

if __name__ == '__main__':
    ddp_processor = DPProcessor(cfg_path="configs/dp_fast_pose.yaml")
    ddp_processor.run()

```
* custom some parameters in *config.yaml*
```yaml
model_name: fast_pose_dp
data:
  train_ann_path: .../data/annotations/person_keypoints_train2017.json
  val_ann_path: .../data/annotations/person_keypoints_val2017.json
  train_img_root: .../data/train2017
  val_img_root: .../data/val2017
  batch_size: 128
  num_workers: 8
  debug: False

model:
  type: pose_resnet_duc
  name: resnet50
  num_joints: 17
  pretrained: True
  reduction: True

optim:
  lr: 0.001
  amp: False
  milestones: [120,160]
  epochs: 180
  gamma: 0.1
val:
  interval: 1
  weight_path: weights

gpus: [4,5,6,7]

```
* run train scripts
```shell script
nohup python main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] EMA(Exponential Moving Average)[**非常不建议使用,训练过程中会有比较大程度的震荡**]
- [x] Mixed Precision Training (supported by apex)
- [x] SELayer
- [x] Sync Batch Normalize
- [x] Person Detector support(by YOLOv5)
- [ ] Test With Person Detector(YOLOv3...)
- [ ] custom data train\test scripts
