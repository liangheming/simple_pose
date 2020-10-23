import numpy as np


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def calc_dist(preds, target, normalize):
    """Calculate normalized distances"""
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, thr=0.5):
    """Calculate accuracy with given input distance."""
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def calc_accuracy(preds, labels):
    """Calculate heatmap accuracy."""
    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    num_joints = preds.shape[1]

    norm = 1.0
    hm_h = preds.shape[2]
    hm_w = preds.shape[3]

    preds, _ = get_max_pred_batch(preds)
    labels, _ = get_max_pred_batch(labels)
    norm = np.ones((preds.shape[0], 2)) * np.array([hm_w, hm_h]) / 10

    dists = calc_dist(preds, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i])
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0
