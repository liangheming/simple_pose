import os
import torch
import json
from tqdm import tqdm
from nets import pose_resnet_dconv
from nets import pose_resnet_duc
from metrics.pose_metrics import BasicKeyPointDecoder, kps_to_dict_, GaussTaylorKeyPointDecoder
from datasets.coco import MSCOCO
from torch.utils.data import DataLoader
import numpy as np


def eval_kps(pd_ann_path="test_gt_kpt.json",
             gt_ann_path="data/annotations/person_keypoints_val2017.json"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_ann_path)
    coco_pd = coco_gt.loadRes(pd_ann_path)
    cocoEval = COCOeval(coco_gt, coco_pd, "keypoints")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]


@torch.no_grad()
def predicts_by_gt():
    vdata = MSCOCO(img_root="data/val2017",
                   ann_path="data/annotations/person_keypoints_val2017.json",
                   debug=False,
                   augment=False,
                   )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model: torch.nn.Module = getattr(pose_resnet_duc, "resnet50")(
        pretrained=False,
        num_classes=17,
        reduction=True
    )
    weights = torch.load("weights/with_reduction/pose_resnet_duc_best.pth", map_location="cpu")['ema']
    weights_info = model.load_state_dict(weights, strict=False)
    print(weights_info)
    device = torch.device("cuda:8")
    model.to(device).eval()
    pbar = tqdm(vloader)
    kps_dict_list = list()
    decoder = GaussTaylorKeyPointDecoder()
    # decoder = BasicKeyPointDecoder()
    for i, (input_tensors, heat_maps, masks, trans_invs, img_ids) in enumerate(pbar):
        input_img = input_tensors.to(device)
        tran_inv = trans_invs.to(device)
        output = model(input_img)
        predicts, scores = decoder(output, tran_inv)
        kps_to_dict_(predicts, scores, img_ids, kps_dict_list)
        # break
    with open("test_gt_kpt.json", "w") as wf:
        json.dump(kps_dict_list, wf)
    eval_kps(pd_ann_path="test_gt_kpt.json")


def gen_data_by_detector():
    import cv2 as cv
    import json
    from pycocotools.coco import COCO
    from detector.yolov5_detector import YOLOv5Detector
    detector = YOLOv5Detector(weights_path="weights/pretrain/coco_yolov5l_best_map.pth",
                              device="cuda:9", conf_thresh=0.25, iou_thresh=0.5)
    coco_keypoints = COCO("data/annotations/person_keypoints_val2017.json")
    all_data = list()
    for entry in tqdm(coco_keypoints.loadImgs(coco_keypoints.getImgIds())):
        file_name = entry['coco_url'].split('/')[-1]
        file_path = os.path.join("data/val2017", file_name)
        width = entry['width']
        height = entry['height']
        box = detector.single_predict(cv.imread(file_path))
        item_data = {
            "file_path": file_path,
            "width": width,
            "height": height,
            "bbox": list()
        }
        if len(box) == 0:
            continue
        box = box.cpu().numpy()
        for box_item in box:
            x1, y1, x2, y2, score = box_item[:-1]
            item_data['bbox'].append((float(x1), float(y1), float(x2), float(y2), float(score)))
        all_data.append(item_data)
    with open("person_detection_ap_59_02.json", "w") as wf:
        json.dump(all_data, wf)


@torch.no_grad()
def predicts_by_pred():
    from datasets.naive_data import MSCOCONoGt
    from nets.pose_hrnet import get_pose_net

    vdata = MSCOCONoGt(img_root="data/val2017",
                       ann_path="data/annotations/COCO_val2017_detections_AP_H_56_person.json",
                       )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model: torch.nn.Module = get_pose_net(
        cfg_path="nets/hrnet_w32.yaml",
        joint_num=17
    )
    # model: torch.nn.Module = getattr(pose_resnet_dconv, "resnet50")(
    #     pretrained=False,
    #     num_classes=17,
    #     reduction=False
    # )
    weights = torch.load("weights/hrnet_pose_dp_best.pth", map_location="cpu")['ema']
    weights_info = model.load_state_dict(weights, strict=False)
    print(weights_info)
    device = torch.device("cuda:8")
    model.to(device).eval()
    pbar = tqdm(vloader)
    kps_predicts = list()
    decoder = GaussTaylorKeyPointDecoder()
    # decoder = BasicKeyPointDecoder()
    for i, (input_tensors, trans_invs, box_info) in enumerate(pbar):
        input_img = input_tensors.to(device)
        tran_inv = trans_invs.to(device)
        output = model(input_img)
        predicts, scores = decoder(output, tran_inv)
        kps_pred = torch.cat([predicts, scores], dim=-1).cpu().numpy()
        for kp, info in zip(kps_pred, box_info):
            kp_list = kp.reshape(-1).tolist()
            item = {
                "kps": kp_list,
                "area": float(info.area),
                "score": info.score,
                "img_id": info.ids
            }
            kps_predicts.append(item)
    with open("predicts_kps_temp.json", "w") as wf:
        json.dump(kps_predicts, wf)
    temp_read_in_and_filter()


def temp_read_in_and_filter(in_vis_thre=0.2, oks_thre=0.9):
    from collections import defaultdict
    from datasets.naive_data import oks_nms
    with open("predicts_kps_temp.json", "r") as rf:
        json_data = json.load(rf)
    kpts = defaultdict(list)

    filter_list = list()
    for kpt in json_data:
        kpts[kpt['img_id']].append(kpt)
    for img_id in kpts.keys():
        img_kpts = kpts[img_id]
        score_list = list()
        area_list = list()
        kpts_list = list()
        for n_p in img_kpts:
            box_score = n_p['score']
            kpt_item = np.array(n_p['kps']).reshape(-1, 3)
            kpt_scores = kpt_item[:, -1]
            valid_mask = kpt_scores > in_vis_thre
            kpt_score = kpt_scores[valid_mask].mean() if valid_mask.sum() > 0 else 0.
            score = box_score * kpt_score
            area = n_p['area']
            score_list.append(score)
            area_list.append(area)
            kpts_list.append(kpt_item)
        kpts_list = np.stack(kpts_list, axis=0)
        score_list = np.array(score_list)
        area_list = np.array(area_list)
        keep = oks_nms(kpts_list, score_list, area_list, oks_thre)
        if len(keep) != 0:
            kpts_list = kpts_list[keep]
            score_list = score_list[keep]
        for kpt_filer, kpt_score in zip(kpts_list, score_list):
            filter_list.append(
                {
                    "image_id": img_id,
                    "score": kpt_score,
                    "category_id": 1,
                    "keypoints": kpt_filer.reshape(-1).tolist()
                }
            )
    with open("filter_kps_predicts.json", 'w') as wf:
        json.dump(filter_list, wf)
    eval_kps(pd_ann_path="filter_kps_predicts.json")


if __name__ == '__main__':
    predicts_by_pred()

# DUC
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.709
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.885
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.781
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.674
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.781
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.768
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.929
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.832
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.722
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.833

# SE+DUC
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.718
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.790
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.683
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.787
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.775
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.932
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.841
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.732
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.836


# DCONV
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.701
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.883
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.772
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.665
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.825
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.715
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.825

# SE+DCONV
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.717
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.890
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.791
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.685
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.776
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.934
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.841
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.733
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.837

# HRNet
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.741
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.807
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.703
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.814
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.795
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.935
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.856
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.750
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.860
