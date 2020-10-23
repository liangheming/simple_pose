import cv2 as cv
import numpy as np
from copy import deepcopy

cv.setNumThreads(0)


class KeyPoints(object):
    EMPTY_VAL = -1

    def __init__(self, img_path, shape, box, keypoints):
        self.img_path = img_path
        self.img = None
        self.shape = shape
        self.box = box
        self.keypoints = keypoints
        self.heat_map = None
        self.mask = None
        self.trans_inv = None

    def load_img(self):
        if self.img is None:
            self.img = cv.imread(self.img_path).astype(np.float32)
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
        invalid_mask = ret_img[..., 0] == self.EMPTY_VAL
        ret_img[invalid_mask, :] = 0
        ret_img = ret_img.astype(np.uint8)
        for idx, bone in enumerate(bones):
            sta_joint, end_joint = self.keypoints[bone[0]], self.keypoints[bone[1]]
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


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = (xmin, ymin, xmax, ymax)
    return bbox


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def affine_transform_batch(pt, t):
    """
    :param pt: [num_gt,2]
    :param t: [2, 3]
    :return:
    """
    new_pt = np.concatenate([pt, np.ones_like(pt[:, [0]])], axis=-1)
    new_pt = np.dot(new_pt, t.T)
    return new_pt[:, :2]


def flip_joints(img, joints_src, joint_pairs):
    _, width = img.shape[:2]
    img_ret = np.fliplr(img)
    joints = joints_src.copy()
    # flip horizontally
    joints[:, 0] = width - joints[:, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0]], joints[pair[1]] = \
            joints[pair[1]], joints[pair[0]].copy()
    return img_ret, joints


def keypoints_heat_map(joints, sigma=2.0, shape=(48, 64), stride=4):
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


class SimpleTransform(object):
    def __init__(self,
                 joint_pairs=None,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 scale=(0.7, 1.3),
                 ratio=(-40, 40)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.joint_pairs = joint_pairs
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, point_info: KeyPoints):
        img = point_info.img
        bbox = point_info.box
        gt_joints = point_info.keypoints
        img_w, _ = point_info.shape
        x1, y1, x2, y2 = bbox
        center, scale = _box_to_center_scale(
            x1, y1, x2 - x1, y2 - y1, self.w_h_ratio)
        scale_ratio = np.random.uniform(self.scale[0], self.scale[1])
        scale = scale * scale_ratio
        rot_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        if self.joint_pairs is not None:
            if np.random.uniform() < 0.5:
                img, gt_joints = flip_joints(img, gt_joints, self.joint_pairs)
                center[0] = img_w - center[0] - 1
        trans = get_affine_transform(center, scale, rot_ratio, self.input_shape)
        trans_inv = get_affine_transform(center, scale, rot_ratio, self.output_shape, inv=1)
        input_img = cv.warpAffine(img, trans, self.input_shape, flags=cv.INTER_LINEAR, borderValue=point_info.EMPTY_VAL)
        point_info.img = input_img
        point_info.trans_inv = trans_inv
        valid_kp_mask = gt_joints[:, 2] > 0
        gt_joints[valid_kp_mask, :2] = affine_transform_batch(gt_joints[valid_kp_mask][:, :2], trans)
        point_info.keypoints = gt_joints
        heat_map, mask = keypoints_heat_map(joints=gt_joints,
                                            sigma=2.0,
                                            shape=self.output_shape,
                                            stride=self.input_shape[0] / self.output_shape[0])
        box = _center_scale_to_box(center, scale)
        point_info.heat_map = heat_map
        point_info.mask = mask
        point_info.box = box
        return point_info
