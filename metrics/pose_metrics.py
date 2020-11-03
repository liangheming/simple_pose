import torch
import sys
import cv2 as cv
import numpy as np
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BasicKeyPointDecoder(object):
    @staticmethod
    def heat_map_to_axis(heat_map: torch.Tensor):
        """
        :param heat_map:[batch,num_joints,h,w]
        :return:
        """
        b, c, h, w = heat_map.shape
        max_val, max_idx = heat_map.view(b, c, -1).max(dim=-1, keepdim=True)
        # [batch,joints,1]
        predicts = max_idx.repeat(1, 1, 2).float()
        predicts[..., 0] = predicts[..., 0] % w
        predicts[..., 1] = (predicts[..., 1] / w).floor()
        predicts = predicts * ((max_val > 0.).repeat(1, 1, 2).float())
        return predicts, max_val

    @torch.no_grad()
    def __call__(self, heat_map: torch.Tensor, trans_inv: torch.Tensor):
        """
        根据变换矩阵将crop中的kps转换到原始图片中的kps
        :param heat_map: [batch,joint_num,h,w]
        :param trans_inv:[batch,2,3]
        :return:
        """
        predicts, max_val = self.heat_map_to_axis(heat_map)
        b, c, h, w = heat_map.shape
        b_idx, c_idx = torch.meshgrid(torch.arange(b), torch.arange(c))
        x_idx, y_idx = predicts[..., 0].long(), predicts[..., 1].long()
        b_idx, c_idx, x_idx, y_idx = b_idx.reshape(-1), c_idx.reshape(-1), x_idx.reshape(-1), y_idx.reshape(-1)
        valid_idx = (x_idx > 1) & (x_idx < w - 1) & (y_idx > 1) & (y_idx < h - 1)
        diff_x = (heat_map[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx], x_idx[valid_idx] + 1]) - \
                 (heat_map[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx], x_idx[valid_idx] - 1])
        diff_x.sign_()
        diff_y = (heat_map[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx] + 1, x_idx[valid_idx]]) - \
                 (heat_map[b_idx[valid_idx], c_idx[valid_idx], y_idx[valid_idx] - 1, x_idx[valid_idx]])
        diff_y.sign_()
        diff = torch.stack([diff_x, diff_y], dim=-1)
        predicts = predicts.view(-1, 2)
        predicts[valid_idx] = predicts[valid_idx] + diff * 0.25
        predicts = predicts.view(b, c, 2)
        xyz = torch.cat([predicts, torch.ones_like(predicts[..., [0]])], dim=-1)
        trans_output = torch.einsum("bcd,bad->bca", xyz, trans_inv)
        return trans_output, max_val


class GaussTaylorKeyPointDecoder(BasicKeyPointDecoder):
    def __init__(self, kernel_size=11, num_joints=17):
        kernel = cv.getGaussianKernel(kernel_size, 0)
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.blur_weights = torch.from_numpy(kernel * kernel.T)[None, None, ...].repeat(num_joints, 1, 1, 1).float()

    @torch.no_grad()
    def __call__(self, heat_map: torch.Tensor, trans_inv: torch.Tensor):
        if heat_map.device != self.blur_weights.device:
            self.blur_weights = self.blur_weights.to(heat_map.device)
        coords, max_val = self.heat_map_to_axis(heat_map)
        # 2. gauss blur
        heat_map_blur = F.conv2d(heat_map, self.blur_weights, bias=None, stride=1, padding=(self.kernel_size - 1) // 2,
                                 groups=self.num_joints)
        b, c, h, w = heat_map.shape
        ori_max, _ = heat_map.view(b, c, -1).max(dim=-1)
        blur_max, _ = heat_map_blur.view(b, c, -1).max(dim=-1)
        heat_map_blur = (heat_map_blur * ori_max[..., None, None] / blur_max[..., None, None]).clamp(min=1e-10).log()
        # 3. calc offsets
        b_idx, c_idx = torch.meshgrid(torch.arange(b), torch.arange(c))
        x_idx, y_idx = coords[..., 0].long(), coords[..., 1].long()
        b_idx, c_idx, x_idx, y_idx = b_idx.reshape(-1), c_idx.reshape(-1), x_idx.reshape(-1), y_idx.reshape(-1)
        valid_mask = (x_idx > 1) & (x_idx < w - 2) & (y_idx > 1) & (y_idx < h - 2)
        v_b_idx, v_c_idx, v_x_idx, v_y_idx = b_idx[valid_mask], c_idx[valid_mask], x_idx[valid_mask], y_idx[valid_mask]
        dx = 0.5 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx + 1] -
                    heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx - 1])
        dy = 0.5 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx] -
                    heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx])
        dxx = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx + 2] -
                      2 * heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx - 2])
        dxy = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx + 1] -
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx + 1] -
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx - 1] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx - 1])
        dyy = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 2, v_x_idx] -
                      2 * heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 2, v_x_idx])
        derivative_valid_mask = dxx * dyy - dxy ** 2 != 0
        dxx_dxy = torch.stack([dxx, dxy], dim=-1)
        dxy_dyy = torch.stack([dxy, dyy], dim=-1)
        v_hessian = torch.stack([dxx_dxy, dxy_dyy], dim=-2)[derivative_valid_mask]
        v_derivative = torch.stack([dx, dy], dim=-1).unsqueeze(-1)[derivative_valid_mask]
        v_hessian_inv = v_hessian.inverse()
        offsets = (-v_hessian_inv @ v_derivative).transpose(1, 2).squeeze(1)
        coords = coords.view(-1, 2)
        valid_mask[valid_mask] = derivative_valid_mask
        coords[valid_mask] = (coords[valid_mask] + offsets).clamp(min=0.)
        coords = coords.view(b, c, 2)
        xyz = torch.cat([coords, torch.ones_like(coords[..., [0]])], dim=-1)
        trans_output = torch.einsum("bcd,bad->bca", xyz, trans_inv)
        return trans_output, max_val


class DarkPoseOriginalKeyPointDecoder(BasicKeyPointDecoder):
    def __init__(self, kernel_size=11):
        self.kernel_size = kernel_size

    @staticmethod
    def taylor(hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py + 1][px + 1] - hm[py - 1][px + 1] - hm[py + 1][px - 1] \
                          + hm[py - 1][px - 1])
            dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy ** 2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    @staticmethod
    def gaussian_blur(hm, kernel):
        border = (kernel - 1) // 2
        batch_size = hm.shape[0]
        num_joints = hm.shape[1]
        height = hm.shape[2]
        width = hm.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(hm[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border: -border, border: -border] = hm[i, j].copy()
                dr = cv.GaussianBlur(dr, (kernel, kernel), 0)
                hm[i, j] = dr[border: -border, border: -border].copy()
                hm[i, j] *= origin_max / np.max(hm[i, j])
        return hm

    def __call__(self, heat_map, trans_inv):
        """
        :param heat_map:
        :param trans_inv:
        :return:
        """
        coords, max_val = self.heat_map_to_axis(heat_map)
        coords = coords.detach().cpu().numpy()
        hm = self.gaussian_blur(heat_map.detach().cpu().numpy(), self.kernel_size)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.taylor(hm[n][p], coords[n][p])
        xyz = np.concatenate([coords, np.ones_like(coords[..., [0]])], axis=-1)
        trans_output = np.einsum("bcd,bad->bca", xyz, trans_inv.cpu().numpy())
        return torch.from_numpy(trans_output), max_val.detach().cpu()


def kps_to_dict_(predicts, scores, img_ids, set_in_list):
    for pd, sc, img_id in zip(predicts, scores, img_ids):
        data = dict()
        data['image_id'] = img_id
        data['score'] = float((sc.mean() + sc.max()).item())
        data['category_id'] = 1
        data['keypoints'] = torch.cat([pd, sc], dim=-1).reshape(-1).cpu().tolist()
        set_in_list.append(data)


def evaluate_map(res_file, ann_file, ann_type='keypoints', silence=True):
    class NullWriter(object):
        def write(self, arg):
            pass

    if silence:
        null_write = NullWriter()
        old_stdout = sys.stdout
        sys.stdout = null_write  # disable output

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_file)

    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if silence:
        sys.stdout = old_stdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = coco_eval.stats[ind]

    return info_str


class HeatMapAcc(object):
    def __init__(self, distance_thresh=0.5, norm_frac=10.):
        self.distance_thresh = distance_thresh
        self.norm_frac = norm_frac

    def __call__(self, predicts, targets):
        """
        :param predicts: [bs, joint_num, h, w]
        :param targets: [bs, joint_num, h, w]
        :return:
        """
        preds, _ = BasicKeyPointDecoder.heat_map_to_axis(predicts)
        labels, _ = BasicKeyPointDecoder.heat_map_to_axis(targets)
        norm = torch.tensor(data=[predicts.shape[-1],
                                  predicts.shape[-2]],
                            dtype=torch.float,
                            device=predicts.device) / self.norm_frac
        valid_mask = (labels[..., 0] > 1) & (labels[..., 1] > 1)
        distance = torch.norm(preds / norm - labels / norm, dim=-1)
        distance[~valid_mask] = -1.
        acc_sum = 0.
        cnt = 0.
        for i in range(distance.shape[1]):
            distance_i = distance.T[i]
            valid_mask_i = valid_mask.T[i]
            if valid_mask_i.sum().item() < 1:
                continue
            joint_acc = (distance_i[valid_mask_i] < self.distance_thresh).sum().float() / (valid_mask_i.sum())
            acc_sum += joint_acc
            cnt += 1
        if cnt > 0:
            return acc_sum / cnt

        return torch.tensor(data=0.)
