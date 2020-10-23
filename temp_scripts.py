import os
import torch
import json
import numpy as np
from tqdm import tqdm
from nets.fastpose import FastPose
from metrics.pose_metrics import heat_map_to_axis
from datasets.coco import MSCOCO
from torch.utils.data import DataLoader


@torch.no_grad()
def predicts_trans():
    vdata = MSCOCO(img_root="/home/huffman/data/val2017",
                   ann_path="/home/huffman/data/annotations/person_keypoints_val2017.json",
                   debug=False,
                   augment=False,
                   )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model = FastPose(reduction=False, pretrained=False)
    weights = torch.load("weights/fast_pose_dp_last.pth", map_location="cpu")['ema']
    weights_info = model.load_state_dict(weights, strict=False)
    device = torch.device("cuda:8")
    model.to(device).eval()
    pbar = tqdm(vloader)
    kps_dict_list = list()
    for i, input_info in enumerate(pbar):
        input_img = input_info['input'].to(device)
        tran_inv = input_info['trans_inv'].to(device)
        kp_info = input_info['box_infos']
        output = model(input_img)
        predicts, scores = heat_map_to_origin_kps(output, tran_inv)
        kps_to_dict_(predicts, scores, kp_info, kps_dict_list)
        # break
    with open("test_gt_kpt.json", "w") as wf:
        json.dump(kps_dict_list, wf)


def heat_map_to_origin_kps(hms: torch.Tensor, trans_inv: torch.Tensor):
    """
    :param hms: [batch,joint_num,h,w]
    :param trans_inv:[batch,2,3]
    :return:
    """
    predicts, max_val = heat_map_to_axis(hms)
    b, c, h, w = hms.shape
    b_idx, c_idx = torch.meshgrid(torch.arange(b), torch.arange(c))
    x_idx, y_idx = predicts[..., 0].long(), predicts[..., 1].long()
    b_idx, c_idx, x_idx, y_idx = b_idx.reshape(-1), c_idx.reshape(-1), x_idx.reshape(-1), y_idx.reshape(-1)
    valid_idx = (x_idx > 1) & (x_idx < w - 1) & (y_idx > 1) & (y_idx < h - 1)
    diff_x = (hms[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx], x_idx[valid_idx] + 1]) - \
             (hms[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx], x_idx[valid_idx] - 1])
    diff_x.sign_()
    diff_y = (hms[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx] + 1, x_idx[valid_idx]]) - \
             (hms[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx] - 1, x_idx[valid_idx]])
    diff_y.sign_()
    diff = torch.stack([diff_x, diff_y], dim=-1)
    predicts = predicts.view(-1, 2)
    predicts[valid_idx] = predicts[valid_idx] + diff * 0.25
    predicts = predicts.view(b, c, 2)
    xyz = torch.cat([predicts, torch.ones_like(predicts[..., [0]])], dim=-1)
    trans_output = torch.einsum("bcd,bad->bca", xyz, trans_inv)
    return trans_output, max_val


def kps_to_dict_(predicts, scores, box_infos, set_in_list):
    for pd, sc, info in zip(predicts, scores, box_infos):
        data = dict()
        data['bbox'] = list(info.box)
        data['image_id'] = int(os.path.splitext(os.path.basename(info.img_path))[0])
        data['score'] = float((sc.mean() + sc.max()).item())
        data['category_id'] = 1
        data['keypoints'] = torch.cat([pd, sc], dim=-1).reshape(-1).cpu().tolist()
        set_in_list.append(data)


def eval_kps():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt_ann_path = "/home/huffman/data/annotations/person_keypoints_val2017.json"
    pd_ann_path = "test_gt_kpt.json"
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
    print(info_str)


@torch.no_grad()
def draw_test():
    import cv2 as cv
    vdata = MSCOCO(img_root="/home/huffman/data/val2017",
                   ann_path="/home/huffman/data/annotations/person_keypoints_val2017.json",
                   debug=False,
                   augment=False,
                   )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model = FastPose(reduction=False, pretrained=False)
    weights = torch.load("weights/temp.pth", map_location="cpu")['ema']
    model.load_state_dict(weights, strict=False)
    device = torch.device("cpu")
    model.to(device).eval()
    pbar = tqdm(vloader)
    for i, input_info in enumerate(pbar):
        input_img = input_info['input'].to(device)
        tran_inv = input_info['trans_inv'].to(device)
        kp_info = input_info['box_infos'][0]
        output = model(input_img)
        kps, score = heat_map_to_origin_kps(output, tran_inv)
        kp_info.keypoints = torch.cat([kps, score], dim=-1)[0].cpu().numpy()
        kp_info.box = None
        kp_info.img = cv.imread(kp_info.img_path)
        ret_img = kp_info.draw_img(bones=vdata.bones,
                                   bones_colors=vdata.bone_colors,
                                   joint_colors=vdata.joint_color)
        if i == 100:
            break
        import uuid
        name = str(uuid.uuid4()).replace('-', "")
        cv.imwrite("temp_imgs/{:s}.jpg".format(name), ret_img)


if __name__ == '__main__':
    predicts_trans()
    eval_kps()
