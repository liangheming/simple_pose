import os
import torch
import cv2 as cv
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from commons.transforms import KeyPoints, SimpleTransform

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]


class MSCOCO(Dataset):
    def __init__(self, img_root, ann_path, crop_shape=(192, 256), debug=False, augment=True):
        self.img_root = img_root
        self.ann_path = ann_path
        self.crop_shape = crop_shape
        self.data_list = list()
        self.check_center = False
        self.__load_in()
        if debug:
            assert debug <= len(self.data_list)
            self.data_list = self.data_list[:debug]
        self.eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.bones = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        self.joint_color = [
            (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
            (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44),
            (195, 150, 66), (2, 249, 42), (195, 135, 43), (105, 70, 66),
            (120, 107, 116), (122, 241, 22), (17, 19, 179), (162, 185, 124), (31, 65, 117)
        ]
        self.bone_colors = [
            (67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90),
            (92, 136, 113), (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145),
            (253, 181, 88), (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53), (153, 207, 15)
        ]

        if augment:
            self.transform = SimpleTransform(joint_pairs=self.joint_pairs,
                                             input_shape=self.crop_shape,
                                             output_shape=(self.crop_shape[0] // 4, self.crop_shape[1] // 4),
                                             scale=(0.7, 1.3),
                                             ratio=(-40, 40))
        else:
            self.transform = SimpleTransform(joint_pairs=None,
                                             input_shape=self.crop_shape,
                                             output_shape=(self.crop_shape[0] // 4, self.crop_shape[1] // 4),
                                             scale=(1.0, 1.0),
                                             ratio=(0, 0))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        info = self.data_list[item].clone().load_img()
        info = self.transform(info)
        # ret_img = info.draw_img(self.bones, self.bone_colors, self.joint_color)
        # mask_img = info.draw_heat_map()
        # mask_img = cv.resize(mask_img, (ret_img.shape[:2][::-1]))
        # ret_img = np.concatenate([ret_img, mask_img], axis=1)
        # import uuid
        # name = str(uuid.uuid4()).replace('-', "")
        # cv.imwrite("{:d}.jpg".format(item), ret_img)
        return info

    def collate_fn(self, batch):
        input_tensors = list()
        heat_maps = list()
        masks = list()
        trans_invs = list()
        for item in batch:
            img = item.img
            # invalid_mask = img[..., 0] == KeyPoints.EMPTY_VAL
            img[img == KeyPoints.EMPTY_VAL] = 0
            norm_img = (img[..., ::-1] / 255.0 - np.array(rgb_mean, dtype=np.float32)) / np.array(rgb_mean,
                                                                                                  dtype=np.float32)
            # norm_img[invalid_mask, :] = 0.
            input_tensors.append(torch.from_numpy(norm_img).permute(2, 0, 1).contiguous())
            heat_map = item.heat_map
            heat_maps.append(torch.from_numpy(heat_map))
            mask = item.mask
            masks.append(torch.from_numpy(mask))
            trans_inv = item.trans_inv
            trans_invs.append(torch.from_numpy(trans_inv))
        input_tensors = torch.stack(input_tensors)
        heat_maps = torch.stack(heat_maps)
        masks = torch.stack(masks)
        trans_invs = torch.stack(trans_invs).float()
        return {
            "input": input_tensors,
            "targets": heat_maps,
            "mask": masks,
            "trans_inv": trans_invs,
            "box_infos": batch
        }

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
                x1, y1, x2, y2 = x1, y1, min(x1 + w, width - 1), min(y1 + h, height - 1)
                if obj['area'] <= 0 or x2 <= x1 or y2 <= y1:
                    print("invalid box")
                    continue
                if obj['num_keypoints'] == 0:
                    continue
                keypoints = np.array(obj['keypoints']).reshape(-1, 3).astype(np.float32)
                keypoints[..., 2] = (keypoints[..., 2] >= 1).astype(np.float32)
                if keypoints[..., 2].sum() < 1.0:
                    continue
                if self.check_center:
                    bbox_center, bbox_area = self._get_box_center_area((x1, y1, x2, y2))
                    kp_center, num_vis = self._get_keypoints_center_count(keypoints)
                    ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                    if (num_vis / 80.0 + 47 / 80.0) > ks:
                        continue
                self.data_list.append(KeyPoints(
                    img_path=file_path,
                    shape=(width, height),
                    box=(x1, y1, x2, y2),
                    keypoints=keypoints
                ))

    @staticmethod
    def _get_box_center_area(bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    @staticmethod
    def _get_keypoints_center_count(keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0] * (keypoints[:, 2] > 0))
        keypoint_y = np.sum(keypoints[:, 1] * (keypoints[:, 2] > 0))
        num = float(np.sum(keypoints[:, 2]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

# if __name__ == '__main__':
#     datasets = MSCOCO(img_root="/home/huffman/data/val2017",
#                       ann_path="/home/huffman/data/annotations/person_keypoints_val2017.json",
#                       debug=100, augment=True)
#     from torch.utils.data.dataloader import DataLoader
#
#     loader = DataLoader(dataset=datasets, collate_fn=datasets.collate_fn, batch_size=4, shuffle=True)
#     for data in loader:
#         input_tensor = data['input']
#         mask = data['mask']
#         print(mask[0])
