import os
import torch
import json
from tqdm import tqdm
from nets import pose_resnet_dconv
from nets import pose_resnet_duc
from metrics.pose_metrics import BasicKeyPointDecoder, kps_to_dict_, GaussTaylorKeyPointDecoder
from datasets.coco import MSCOCO
from torch.utils.data import DataLoader


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
                   ann_path="annotations/person_keypoints_val2017.json",
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
def predicts_by_detector():
    from datasets.naive_data import NaiveDataset
    vdata = NaiveDataset(
        json_path="person_detection_ap_59_02.json",
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
    for i, (input_tensors, trans_invs, img_ids) in enumerate(pbar):
        input_img = input_tensors.to(device)
        tran_inv = trans_invs.to(device)
        output = model(input_img)
        predicts, scores = decoder(output, tran_inv)
        kps_to_dict_(predicts, scores, img_ids, kps_dict_list)
        # break
    with open("test_detector_kpt.json", "w") as wf:
        json.dump(kps_dict_list, wf)
    eval_kps(pd_ann_path="test_detector_kpt.json")


@torch.no_grad()
def predicts_by_pred():
    from datasets.naive_data import MSCOCONoGt
    vdata = MSCOCONoGt(img_root="data/val2017",
                       ann_path="data/annotations/person_keypoints_val2017.json",
                       )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model: torch.nn.Module = getattr(pose_resnet_dconv, "resnet50")(
        pretrained=False,
        num_classes=17,
        reduction=False
    )
    weights = torch.load("weights/without_reduction/fast_pose_dp_dconv_best.pth", map_location="cpu")['ema']
    weights_info = model.load_state_dict(weights, strict=False)
    print(weights_info)
    device = torch.device("cuda:8")
    model.to(device).eval()
    pbar = tqdm(vloader)
    kps_dict_list = list()
    decoder = GaussTaylorKeyPointDecoder()
    # decoder = BasicKeyPointDecoder()
    for i, (input_tensors, trans_invs, img_ids) in enumerate(pbar):
        input_img = input_tensors.to(device)
        tran_inv = trans_invs.to(device)
        output = model(input_img)
        predicts, scores = decoder(output, tran_inv)
        kps_to_dict_(predicts, scores, img_ids, kps_dict_list)
        # break
    with open("test_pred_kpt.json", "w") as wf:
        json.dump(kps_dict_list, wf)
    eval_kps(pd_ann_path="test_pred_kpt.json")


if __name__ == '__main__':
    predicts_by_pred()
# SE+DUC
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.734
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.923
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.810
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.706
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.777
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.768
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.932
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.831
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.734
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.818

# DUC+GaussTaylor
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.726
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.923
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.801
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.698
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.930
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.823
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.724
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.813

# DCONV
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.714
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.913
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.789
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.688
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.760
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.752
# Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928
# Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.815
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.719
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.802
