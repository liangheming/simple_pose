import os
import json
import torch
import cv2 as cv
import numpy as np
from torch.utils.data.dataset import Dataset
from commons.joint_utils import box_to_center_scale, get_affine_transform
from copy import deepcopy

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]


class KeyPointItem(object):
    def __init__(self, img_path, box, score):
        self.img_path = img_path
        self.img = None
        self.box = box
        self.score = score
        self.scale = None
        self.center = None
        self.area = None

    def load_img(self):
        if self.img is None:
            self.img = cv.imread(self.img_path)
        return self

    def clone(self):
        return deepcopy(self)


class BasicTransform(object):
    def __init__(self,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]

    def __call__(self, joint_info: KeyPointItem):
        img = joint_info.img
        bbox = joint_info.box
        x1, y1, x2, y2 = bbox
        center, scale = box_to_center_scale(
            x1, y1, x2 - x1, y2 - y1, self.w_h_ratio)
        img_trans, _ = get_affine_transform(center, scale, 0, self.input_shape)
        _, joint_trans_inv = get_affine_transform(center, scale, 0, self.output_shape)
        input_img = cv.warpAffine(img, img_trans, self.input_shape, flags=cv.INTER_LINEAR)
        joint_info.img = input_img
        joint_info.trans_inv = joint_trans_inv
        joint_info.center = center
        joint_info.scale = scale
        joint_info.area = scale[0] * scale[1]
        return joint_info


class MSCOCONoGt(Dataset):
    def __init__(self,
                 img_root,
                 ann_path,
                 crop_shape=(192, 256), debug=False):
        super(MSCOCONoGt, self).__init__()
        self.img_root = img_root
        self.ann_path = ann_path
        self.crop_shape = crop_shape
        self.data_list = list()
        self.__load_in()
        if debug:
            assert debug <= len(self.data_list)
            self.data_list = self.data_list[:debug]
        self.transform = BasicTransform(input_shape=self.crop_shape,
                                        output_shape=(self.crop_shape[0] // 4, self.crop_shape[1] // 4),
                                        )

    def __getitem__(self, item):
        info = self.data_list[item].clone().load_img()
        info = self.transform(info)
        return info

    def __len__(self):
        return len(self.data_list)

    def __load_in(self):
        with open(self.ann_path, 'r') as rf:
            json_data = json.load(rf)
        for item in json_data:
            x, y, w, h = item['bbox']
            image_id = item['image_id']
            category_id = item['category_id']
            if category_id != 1:
                continue
            score = item['score']
            image_name = "{0:012d}.jpg".format(image_id)
            file_path = os.path.join(self.img_root, image_name)
            kps = KeyPointItem(
                img_path=file_path,
                box=[x, y, x + w, y + h],
                score=score
            )
            self.data_list.append(kps)

    @staticmethod
    def collate_fn(batch):
        input_tensors = list()
        trans_invs = list()
        for item in batch:
            img = item.img
            item.ids = int(os.path.splitext(os.path.basename(item.img_path))[0])
            norm_img = (img[..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32))
            input_tensors.append(torch.from_numpy(norm_img).permute(2, 0, 1).contiguous())
            trans_inv = item.trans_inv
            trans_invs.append(torch.from_numpy(trans_inv))
        input_tensors = torch.stack(input_tensors)
        trans_invs = torch.stack(trans_invs).float()
        return input_tensors, trans_invs, batch


def oks_iou(pick_kps, candi_kps, pick_area, candi_area, sigmas=None, in_vis_thresh=None):
    """
    :param pick_kps:[kp_num,3]
    :param candi_kps:[gt_num,kp_num,3]
    :param pick_area:
    :param candi_area:
    :param sigmas:
    :param in_vis_thresh:
    :return:
    """
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) \
                 / 10.0
    var = (sigmas * 2) ** 2
    xg = pick_kps[:, 0]
    yg = pick_kps[:, 1]
    vg = pick_kps[:, 2]
    xds = candi_kps[..., 0]
    yds = candi_kps[..., 1]
    vds = candi_kps[..., 2]

    dx = xds - xg
    dy = yds - yg
    # [gt_num,17]
    e = (dx ** 2 + dy ** 2) / var / ((pick_area + candi_area)[:, None] / 2 + 1e-12) / 2
    vd_vis = np.ones_like(vds, dtype=np.float32)
    if in_vis_thresh is not None:
        vg_vis = np.tile((vg > in_vis_thresh)[None, :], (vds.shape[0], 1))
        vd_vis = ((vds > in_vis_thresh) & vg_vis).astype(np.float32)
    ious = (np.exp(-e) * vd_vis).sum(-1) / (vd_vis.sum(-1) + 1e-12)
    return ious


def oks_nms(kps, scores, areas, thresh, sigmas=None, in_vis_thresh=None):
    """
    :param kps:[gt_num,kp_num,3] (x1,y1,conf)
    :param scores:[gt_num]
    :param areas:[gt_num]
    :param thresh:
    :param sigmas:
    :param in_vis_thresh:
    :return:
    """
    order = scores.argsort()[::-1]
    keep = list()
    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]
        if order.size == 0:
            break
        oks_ovr = oks_iou(kps[pick_idx], kps[order], areas[pick_idx], areas[order], sigmas, in_vis_thresh)
        order = order[oks_ovr <= thresh]
    return keep


def oks_iou_ori(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


# if __name__ == '__main__':
#     datasets = MSCOCONoGt(img_root="/home/huffman/data/val2017",
#                           ann_path="/home/huffman/data/annotations/COCO_val2017_detections_AP_H_56_person.json",
#                           )
#     from torch.utils.data.dataloader import DataLoader
#
#     #
#     loader = DataLoader(dataset=datasets, batch_size=4, shuffle=True, collate_fn=datasets.collate_fn)
#     for input_tensors, trans_invs, kps in loader:
#         # print(input_tensors.shape, trans_invs.shape)
#         print(kps[0].ids)
#         break
# print("#" * 50)
# for input_tensors, heat_maps, masks, trans_invs, img_ids in loader:
#     print(masks[0])
if __name__ == '__main__':
    kps = np.random.random(size=(17, 3))
    kps[:, :2] = kps[:, :2] * 100
    candi_kps = np.random.random(size=(10, 17, 3))
    candi_kps[..., :2] = candi_kps[..., :2] * 100
    area = np.random.random((1,)) * 50
    candi_area = np.random.random((10,)) * 50
    iou = oks_iou(kps, candi_kps, area, candi_area, in_vis_thresh=0.1)
    iou_ori = oks_iou_ori(kps.reshape(-1), candi_kps.reshape(10, -1), area, candi_area, in_vis_thre=0.1)
