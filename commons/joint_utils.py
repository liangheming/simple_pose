import random
import cv2 as cv
import numpy as np


def box_crop(bbox, img_width, img_ht):
    """Add dpg for data augmentation, including random crop and random sample."""
    path_scale = random.uniform(0, 1)
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]

    if path_scale > 0.85:
        ratio = ht / width
        if width < ht:
            patch_width = path_scale * width
            patch_ht = patch_width * ratio
        else:
            patch_ht = path_scale * ht
            patch_width = patch_ht / ratio

        xmin = bbox[0] + random.uniform(0, 1) * (width - patch_width)
        ymin = bbox[1] + random.uniform(0, 1) * (ht - patch_ht)
        xmax = xmin + patch_width + 1
        ymax = ymin + patch_ht + 1
    else:
        xmin = max(1, min(bbox[0] + np.random.normal(-0.0142, 0.1158) * width, img_width - 3))
        ymin = max(1, min(bbox[1] + np.random.normal(0.0043, 0.068) * ht, img_ht - 3))
        xmax = min(max(xmin + 2, bbox[2] + np.random.normal(0.0154, 0.1337) * width), img_width - 3)
        ymax = min(max(ymin + 2, bbox[3] + np.random.normal(-0.0013, 0.0711) * ht), img_ht - 3)

    bbox[0] = xmin
    bbox[1] = ymin
    bbox[2] = xmax
    bbox[3] = ymax

    return bbox


def box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1.0
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


def center_scale_to_box(center, scale):
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


def affine_transform_batch(joints, t):
    """
    :param joints: [num_gt,3]
    :param t: [2, 3]
    :return:
    """
    joints = joints.copy()
    valid_kp_mask = joints[:, 2] > 0
    joints_sig = np.concatenate([joints[valid_kp_mask, :2], np.ones_like(joints[valid_kp_mask, 0:1])], axis=-1)
    new_joints = np.dot(joints_sig, t.T)
    joints[valid_kp_mask, :2] = new_joints
    return joints


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


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32)):
    """
    获得转换仿射变换的矩阵和逆矩阵
    :param center: 中心
    :param scale: 宽高
    :param rot: 旋转角度
    :param output_size: 输出图片尺寸
    :param shift:
    :return:
    """
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
    trans_inv = cv.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv.getAffineTransform(np.float32(src), np.float32(dst))
    return trans, trans_inv


