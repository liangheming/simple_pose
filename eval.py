import os
import torch
import json
from tqdm import tqdm
from nets import pose_resnet_dconv
from nets import pose_resnet_duc
from metrics.pose_metrics import BasicKeyPointDecoder, kps_to_dict_, GaussTaylorKeyPointDecoder
from datasets.coco import MSCOCO
from torch.utils.data import DataLoader


@torch.no_grad()
def predicts_trans():
    vdata = MSCOCO(img_root="data/val2017",
                   ann_path="data/annotations/person_keypoints_val2017.json",
                   debug=False,
                   augment=False,
                   )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=2,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    model: torch.nn.Module = getattr(pose_resnet_dconv, "resnet50")(
        pretrained=False,
        num_classes=17,
    )
    weights = torch.load("weights/without_reduction/fast_pose_dp_dconv_best.pth", map_location="cpu")['ema']
    weights_info = model.load_state_dict(weights, strict=False)
    print(weights_info)
    device = torch.device("cuda:8")
    model.to(device).eval()
    pbar = tqdm(vloader)
    kps_dict_list = list()
    # decoder = GaussTaylorKeyPointDecoder()
    decoder = BasicKeyPointDecoder()
    for i, (input_tensors, heat_maps, masks, trans_invs, img_ids) in enumerate(pbar):
        input_img = input_tensors.to(device)
        tran_inv = trans_invs.to(device)
        output = model(input_img)
        predicts, scores = decoder(output, tran_inv)
        kps_to_dict_(predicts, scores, img_ids, kps_dict_list)
        # break
    with open("test_gt_kpt.json", "w") as wf:
        json.dump(kps_dict_list, wf)


def eval_kps():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt_ann_path = "data/annotations/person_keypoints_val2017.json"
    pd_ann_path = "test_gt_kpt.json"
    coco_gt = COCO(gt_ann_path)
    coco_pd = coco_gt.loadRes(pd_ann_path)
    cocoEval = COCOeval(coco_gt, coco_pd, "keypoints")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]


if __name__ == '__main__':
    predicts_trans()
    eval_kps()
