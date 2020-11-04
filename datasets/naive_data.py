import os
import json
import torch
import cv2 as cv
import numpy as np
from torch.utils.data.dataset import Dataset
from commons.transforms import KeyPoints
from commons.joint_utils import box_to_center_scale, get_affine_transform
from pycocotools.coco import COCO

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]


class BasicTransform(object):
    def __init__(self,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]

    def __call__(self, joint_info: KeyPoints):
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
        return joint_info


class NaiveDataset(Dataset):
    def __init__(self, json_path, crop_shape=(192, 256), debug=False, score_thresh=0.001):
        super(NaiveDataset, self).__init__()
        self.json_path = json_path
        self.crop_shape = crop_shape
        self.data_list = list()
        self.score_thresh = score_thresh
        self.__load_in()
        if debug:
            assert debug <= len(self.data_list)
            self.data_list = self.data_list[:debug]
        self.transform = BasicTransform(
            input_shape=self.crop_shape,
            output_shape=(self.crop_shape[0] // 4, self.crop_shape[1] // 4),
        )

    def __getitem__(self, item):
        info = self.data_list[item].clone().load_img()
        info = self.transform(info)
        # info.box = None
        # ret_img = info.draw_img(self.bones, self.bone_colors, self.joint_color)
        # mask_img = info.draw_heat_map()
        # mask_img = cv.resize(mask_img, (ret_img.shape[:2][::-1]))
        # ret_img = np.concatenate([info.img, ret_img, mask_img], axis=1)
        # import uuid
        # name = str(uuid.uuid4()).replace('-', "")
        # cv.imwrite("{:s}_{:d}.jpg".format(name, item), ret_img)
        return info

    def __len__(self):
        return len(self.data_list)

    def __load_in(self):
        with open(self.json_path, 'r') as rf:
            json_data = json.load(rf)
        for item in json_data:
            file_path = item['file_path']
            width = item['width']
            height = item['height']
            bbox = item['bbox']
            if len(bbox) == 0:
                continue
            for box_item in bbox:
                x1, y1, x2, y2, score = box_item
                if score < self.score_thresh:
                    continue
                self.data_list.append(KeyPoints(
                    img_path=file_path,
                    shape=(width, height),
                    box=[x1, y1, x2, y2],
                    joints=None
                ))

    @staticmethod
    def collate_fn(batch):
        input_tensors = list()
        trans_invs = list()
        img_ids = list()
        for item in batch:
            img = item.img
            img_ids.append(int(os.path.splitext(os.path.basename(item.img_path))[0]))
            # norm_img = (img[..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32)) / np.array(
            #     rgb_std, dtype=np.float32)
            norm_img = (img[..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32))
            input_tensors.append(torch.from_numpy(norm_img).permute(2, 0, 1).contiguous())
            trans_inv = item.trans_inv
            trans_invs.append(torch.from_numpy(trans_inv))
        input_tensors = torch.stack(input_tensors)
        trans_invs = torch.stack(trans_invs).float()
        return input_tensors, trans_invs, img_ids


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
        coco_keypoints = COCO(self.ann_path)
        for entry in coco_keypoints.loadImgs(coco_keypoints.getImgIds()):
            file_name = entry['coco_url'].split('/')[-1]
            file_path = os.path.join(self.img_root, file_name)
            ann_ids = coco_keypoints.getAnnIds(imgIds=entry['id'], iscrowd=False)
            objs = coco_keypoints.loadAnns(ann_ids)
            width = entry['width']
            height = entry['height']
            for obj in objs:
                x1, y1, w, h = obj['bbox']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(x1 + max(0, w - 1), width - 1), min(y1 + max(0, h - 1), height - 1)
                if obj['area'] <= 0 or x2 <= x1 or y2 <= y1:
                    print("invalid box")
                    continue
                self.data_list.append(KeyPoints(
                    img_path=file_path,
                    shape=(width, height),
                    box=[x1, y1, x2, y2],
                    joints=None
                ))

    @staticmethod
    def collate_fn(batch):
        input_tensors = list()
        trans_invs = list()
        img_ids = list()
        for item in batch:
            img = item.img
            img_ids.append(int(os.path.splitext(os.path.basename(item.img_path))[0]))
            norm_img = (img[..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32))
            input_tensors.append(torch.from_numpy(norm_img).permute(2, 0, 1).contiguous())
            trans_inv = item.trans_inv
            trans_invs.append(torch.from_numpy(trans_inv))
        input_tensors = torch.stack(input_tensors)
        trans_invs = torch.stack(trans_invs).float()
        return input_tensors, trans_invs, img_ids


if __name__ == '__main__':
    datasets = MSCOCONoGt(img_root="/home/huffman/data/val2017",
                          ann_path="/home/huffman/data/annotations/person_keypoints_val2017.json",
                          )
    from torch.utils.data.dataloader import DataLoader

    #
    loader = DataLoader(dataset=datasets, batch_size=4, shuffle=True, collate_fn=datasets.collate_fn)
    for input_tensors, trans_invs, img_ids in loader:
        print(input_tensors.shape, trans_invs.shape)
# print("#" * 50)
# for input_tensors, heat_maps, masks, trans_invs, img_ids in loader:
#     print(masks[0])
