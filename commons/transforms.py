from copy import deepcopy
from commons.joint_utils import *

cv.setNumThreads(0)


class KeyPoints(object):
    EMPTY_VAL = 0

    def __init__(self, img_path, shape, box, joints):
        self.img_path = img_path
        self.img = None
        self.shape = shape
        self.box = box
        self.joints = joints
        self.heat_map = None
        self.mask = None
        self.trans_inv = None

    def load_img(self):
        if self.img is None:
            self.img = cv.imread(self.img_path)
            assert len(self.img.shape) == 3, "make sure channel num is 3"
            if self.shape is not None:
                assert self.shape[0] == self.img.shape[1] and self.shape[1] == self.img.shape[0]
            else:
                self.shape = [self.img.shape[1], self.img.shap[0]]
        return self

    def clone(self):
        return deepcopy(self)

    def draw_img(self, bones, bones_colors, joint_colors):
        assert self.img is not None
        ret_img = self.img.copy()
        # invalid_mask = ret_img[..., 0] == self.EMPTY_VAL
        # ret_img[invalid_mask, :] = 0
        # ret_img = ret_img.astype(np.uint8)
        for idx, bone in enumerate(bones):
            sta_joint, end_joint = self.joints[bone[0]], self.joints[bone[1]]
            if sta_joint[2] != 0:
                x, y = sta_joint[:2]
                cv.circle(ret_img, center=(int(x), int(y)), radius=2, color=joint_colors[bone[0]], thickness=-1)
            if end_joint[2] != 0:
                x, y = end_joint[:2]
                cv.circle(ret_img, center=(int(x), int(y)), radius=2, color=joint_colors[bone[1]], thickness=-1)
            if sta_joint[2] != 0 and end_joint[2] != 0:
                x1, y1 = sta_joint[:2]
                x2, y2 = end_joint[:2]
                cv.line(ret_img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=bones_colors[idx], thickness=2)
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            cv.rectangle(ret_img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)
        return ret_img

    def draw_heat_map(self):
        assert self.img is not None and self.heat_map is not None
        merge_map = (self.heat_map * self.mask[:, None, None]).max(axis=0)
        ret_map = (merge_map * 255).astype(np.uint8)
        ret_map = cv.cvtColor(ret_map, cv.COLOR_GRAY2BGR)
        return ret_map


class BasicSimpleTransform(object):
    def __init__(self,
                 joint_pairs=None,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 scale=(0.7, 1.3),
                 ratio=(-40, 40),
                 rand_crop=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.joint_pairs = joint_pairs
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]
        self.scale = scale
        self.ratio = ratio
        self.rand_crop = rand_crop

    @staticmethod
    def get_heat_map(joints, sigma=2.0, shape=(48, 64), stride=4):
        """
        :param joints: [gt_num,3]
        :param sigma:
        :param shape: [w,h]
        :param stride: feature map stride
        :return:
        """
        num_joints = joints.shape[0]
        weights = joints[:, 2].copy()
        targets = np.zeros((num_joints, shape[1], shape[0]), dtype=np.float32)
        tmp_size = sigma * 3

        for i in range(num_joints):
            mu_x = int(joints[i, 0] / stride + 0.5)
            mu_y = int(joints[i, 1] / stride + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= shape[0] or ul[1] >= shape[1] or br[0] < 0 or br[1] < 0:
                weights[i] = 0.
                continue
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
            g_x = max(0, -ul[0]), min(br[0], shape[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], shape[1]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], shape[0])
            img_y = max(0, ul[1]), min(br[1], shape[1])
            v = weights[i]
            if v > 0.5:
                targets[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return targets, weights

    def __call__(self, joint_info: KeyPoints):
        img = joint_info.img
        bbox = joint_info.box
        gt_joints = joint_info.joints
        img_w, img_h = joint_info.shape
        if self.rand_crop:
            bbox = box_crop(bbox, img_w, img_h)
        x1, y1, x2, y2 = bbox
        center, scale = box_to_center_scale(
            x1, y1, x2 - x1, y2 - y1, self.w_h_ratio)
        scale_ratio = np.random.uniform(self.scale[0], self.scale[1])
        scale = scale * scale_ratio
        rot_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        # 进行随机水平翻转
        if self.joint_pairs is not None:
            if np.random.uniform() < 0.5:
                img, gt_joints = flip_joints(img, gt_joints, self.joint_pairs)
                center[0] = img_w - center[0] - 1
        # 获得仿宿变换的矩阵
        img_trans, _ = get_affine_transform(center, scale, rot_ratio, self.input_shape)
        _, joint_trans_inv = get_affine_transform(center, scale, rot_ratio, self.output_shape)
        input_img = cv.warpAffine(img, img_trans, self.input_shape, flags=cv.INTER_LINEAR)
        joint_info.img = input_img
        joint_info.trans_inv = joint_trans_inv
        gt_joints = affine_transform_batch(gt_joints, img_trans)
        joint_info.joints = gt_joints
        heat_map, mask = self.get_heat_map(gt_joints, sigma=2.0, shape=self.output_shape)
        joint_info.box = center_scale_to_box(center, scale)
        joint_info.heat_map = heat_map
        joint_info.mask = mask
        return joint_info


class RefineSimpleTransform(object):
    def __init__(self,
                 joint_pairs=None,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 scale=(0.7, 1.3),
                 ratio=(-40, 40),
                 rand_crop=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.joint_pairs = joint_pairs
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]
        self.scale = scale
        self.ratio = ratio
        self.rand_crop = rand_crop

    @staticmethod
    def get_heat_map(joints, sigma=2.0, shape=(48, 64)):
        """
        :param joints:[joint_num,3] (x,y,vis)
        :param sigma:
        :param shape:
        :return:
        """
        num_joints = joints.shape[0]
        weights = joints[:, 2].copy()
        targets = np.zeros((num_joints, shape[1], shape[0]), dtype=np.float32)
        tmp_size = sigma * 3
        for i in range(num_joints):
            mu_x, mu_y = joints[i, :2]
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= shape[0] or ul[1] >= shape[1] or br[0] < 0 or br[1] < 0:
                weights[i] = 0.
                continue
            v = weights[i]
            if v > 0.5:
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
                xy = np.stack([x, y], axis=-1)
                targets[i] = np.exp(-np.sum((xy - np.array([mu_x, mu_y])) ** 2, axis=-1) / (2 * sigma ** 2))
        return targets, weights

    def __call__(self, joint_info: KeyPoints):
        img = joint_info.img
        bbox = joint_info.box
        gt_joints = joint_info.joints.copy()
        img_w, img_h = joint_info.shape
        if self.rand_crop:
            bbox = box_crop(bbox, img_w, img_h)
        x1, y1, x2, y2 = bbox
        center, scale = box_to_center_scale(
            x1, y1, x2 - x1, y2 - y1, self.w_h_ratio)
        scale_ratio = np.random.uniform(self.scale[0], self.scale[1])
        scale = scale * scale_ratio
        rot_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        # 进行随机水平翻转
        if self.joint_pairs is not None:
            if np.random.uniform() < 0.5:
                img, gt_joints = flip_joints(img, gt_joints, self.joint_pairs)
                center[0] = img_w - center[0] - 1
        # 获得仿宿变换的矩阵
        img_trans, _ = get_affine_transform(center, scale, rot_ratio, self.input_shape)
        joint_trans, joint_trans_inv = get_affine_transform(center, scale, rot_ratio, self.output_shape)
        input_img = cv.warpAffine(img, img_trans, self.input_shape, flags=cv.INTER_LINEAR)
        joint_info.img = input_img
        joint_info.trans_inv = joint_trans_inv
        joint_info.joints = affine_transform_batch(gt_joints, img_trans)
        gt_joints = affine_transform_batch(gt_joints, joint_trans)
        heat_map, mask = self.get_heat_map(gt_joints, sigma=2.0, shape=self.output_shape)
        joint_info.box = center_scale_to_box(center, scale)
        joint_info.heat_map = heat_map
        joint_info.mask = mask
        return joint_info
