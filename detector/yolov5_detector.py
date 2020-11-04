import torch
import cv2 as cv
import numpy as np
from torch import nn
from detector.nets.yolov5 import YOLOv5
from torchvision.ops.boxes import nms


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction,
                        conf_thresh=0.1,
                        iou_thresh=0.6,
                        merge=False,
                        agnostic=False,
                        multi_label=True,
                        max_det=300):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    xc = prediction[..., 4] > conf_thresh  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    redundant = True  # require redundant detections
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thresh).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thresh]

        # Filter by class

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thresh  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]

    return output


class ScalePadding(object):
    """
    等比缩放长边至指定尺寸，padding短边部分
    """

    def __init__(self, target_size=(640, 640),
                 padding_val=(114, 114, 114),
                 minimum_rectangle=False,
                 scale_up=True, **kwargs):
        super(ScalePadding, self).__init__(**kwargs)
        self.p = 1
        self.new_shape = target_size
        self.padding_val = padding_val
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up

    def make_border(self, img: np.ndarray):
        # h,w
        shape = img.shape[:2]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)
        r = min(self.new_shape[1] / shape[0], self.new_shape[0] / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[0] - new_unpad[0], self.new_shape[1] - new_unpad[1]
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)

        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=self.padding_val)
        return img, ratio, (left, top)


class MConv2D(nn.Module):
    def __init__(self, weights, bias, stride=1, padding=0):
        super(MConv2D, self).__init__()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = nn.functional.conv2d(x, self.weights, self.bias, stride=self.stride, padding=self.padding)
        return x


class YOLOv5Detector(object):
    def __init__(self, weights_path,
                 num_cls=80,
                 scale_name="l",
                 scale_size=(640, 640),
                 device="cpu",
                 iou_thresh=0.6,
                 conf_thresh=0.001,
                 slice_idx=0):
        self.device = torch.device(device)
        self.model = YOLOv5(scale_name=scale_name, num_cls=num_cls)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu")['ema'])
        self.transform = ScalePadding(target_size=scale_size,
                                      minimum_rectangle=True,
                                      padding_val=(114, 114, 114))

        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        if slice_idx >= 0:
            new_heads = list()
            output_num = self.model.head.output_num
            for conv in self.model.head.heads:
                oc, in_c, k1, k2 = conv.weight.shape
                weight = conv.weight.view(-1, output_num, in_c, k1, k2)[:, [0, 1, 2, 3, 4, 5 + slice_idx], :, :, :]
                weight = weight.view(-1, in_c, k1, k2)
                bias = conv.bias.view(-1, output_num)[:, [0, 1, 2, 3, 4, 5 + slice_idx]].view(-1)
                new_conv = MConv2D(weight, bias)
                new_heads.append(new_conv)
            self.model.head.heads = nn.ModuleList(new_heads)
            self.model.head.num_cls = 1
            self.model.head.output_num = 6
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def single_predict(self, img):
        """
        :param img:
        :return:
        """
        img, ratio, (left, top) = self.transform.make_border(img)
        h, w = img.shape[:2]
        img_out = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).div(255.0).to(self.device)
        predicts = self.model(img_out)
        box = non_max_suppression(predicts,
                                  multi_label=True,
                                  iou_thresh=self.iou_thresh,
                                  conf_thresh=self.conf_thresh,
                                  merge=True)[0]
        if box is None:
            return []
        clip_coords(box, (h, w))
        # x1,y1,x2,y2,score,cls_id
        box[:, [0, 2]] = (box[:, [0, 2]] - left) / ratio[0]
        box[:, [1, 3]] = (box[:, [1, 3]] - top) / ratio[1]
        return box
