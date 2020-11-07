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
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.701
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.883
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.772
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.665
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.825
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.715
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.825
```
### DUC(上采样Conv+PixelShuffle)的performance)
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.709
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.885
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.781
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.674
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.781
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.768
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.929
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.832
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.722
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.833
```
### SELayer+DUC(上采样Conv+PixelShuffle)的performance
**模型初始化时的reduction为True**
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.718
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.790
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.683
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.787
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.775
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.932
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.841
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.732
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.836
```

### SELayer+DConv(上采样使用转置卷积)的performance
**模型初始化时的reduction为True**
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.717
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.890
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.791
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.685
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.776
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.934
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.841
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.733
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.837
```
### HRNet W32 performance
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.741
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.807
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.703
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.814
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.795
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.935
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.856
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.750
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.860
```

### SE_DUC+YOLOv5(目标检测使用重写的YOLOv5)
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.723
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.903
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.794
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.689
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.787
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.780
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.940
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.845
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.739
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.839
```
[目标检测权重以及姿态估计的权重](https://pan.baidu.com/s/1O4u1wOklZOj-OVYivpRX1w) 云盘密码:e5f9



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
- [x] Test With Person Detector(YOLOv3/v4/v5...)
- [ ] custom data train\test scripts
- [ ] video demo\friendly API
