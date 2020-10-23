import torch


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


def heat_map_to_origin_kps(hms: torch.Tensor, trans_inv: torch.Tensor):
    """
    根据变换矩阵将crop中的kps转换到原始图片中的kps
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
        preds, _ = heat_map_to_axis(predicts)
        labels, _ = heat_map_to_axis(targets)
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


if __name__ == '__main__':
    a = torch.rand(size=(32, 17, 64, 48))
    b = torch.rand(size=(32, 17, 64, 48))
    from commons.temp_utils import calc_accuracy, get_max_pred_batch

    acc_func = HeatMapAcc()
    acc0 = calc_accuracy(a, b)
    acc1 = acc_func(a, b)
    print((get_max_pred_batch(a.numpy())[1] != heat_map_to_axis(a)[1].numpy()).sum())
    print((get_max_pred_batch(b.numpy())[1] != heat_map_to_axis(b)[1].numpy()).sum())
    print(acc0, acc1)
