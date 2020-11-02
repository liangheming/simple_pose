import os
import yaml
import torch
import json
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from nets import pose_resnet_duc
from nets import pose_resnet_dconv
from torch.cuda import amp
from metrics.pose_metrics import HeatMapAcc, BasicKeyPointDecoder, kps_to_dict_, evaluate_map
from datasets.coco import MSCOCO
from commons.model_utils import rand_seed, AverageLogger
from torch.optim.adam import Adam
from commons.optims_utils import IterWarmUpCosineDecayMultiStepLRAdjust

rand_seed(1024)


class DPProcessor(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.val_cfg)
        self.tdata = MSCOCO(img_root=self.data_cfg['train_img_root'],
                            ann_path=self.data_cfg['train_ann_path'],
                            debug=self.data_cfg['debug'],
                            augment=True,
                            )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  shuffle=True)
        self.vdata = MSCOCO(img_root=self.data_cfg['val_img_root'],
                            ann_path=self.data_cfg['val_ann_path'],
                            debug=False,
                            augment=False,
                            )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collate_fn,
                                  shuffle=False
                                  )
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata))
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))
        model: torch.nn.Module = getattr(eval(self.model_cfg['type']), self.model_cfg['name'])(
            pretrained=self.model_cfg['pretrained'],
            num_classes=self.model_cfg['num_joints'],
            reduction=self.model_cfg['reduction']
        )
        self.scaler = amp.GradScaler(enabled=True) if self.optim_cfg['amp'] else None
        self.optimizer = Adam(
            model.parameters(), lr=self.optim_cfg['lr']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.optim_cfg['milestones'],
            gamma=self.optim_cfg['gamma']
        )
        # self.lr_scheduler = IterWarmUpCosineDecayMultiStepLRAdjust(
        #     init_lr=self.optim_cfg['lr'],
        #     milestones=self.optim_cfg['milestones'],
        #     warm_up_epoch=1,
        #     iter_per_epoch=len(self.tloader),
        #     epochs=self.optim_cfg['epochs']
        # )

        assert torch.cuda.is_available(), "training only support cuda"
        assert torch.cuda.device_count() >= len(self.cfg['gpus']), "not have enough gpus"
        self.inp_device = torch.device("cuda:{:d}".format(self.cfg['gpus'][0]))
        self.out_device = torch.device("cuda:{:d}".format(self.cfg['gpus'][-1]))
        model.to(self.inp_device)
        self.model = nn.DataParallel(
            model, device_ids=self.cfg['gpus'], output_device=self.out_device)
        # self.ema = ModelEMA(self.model)
        self.creterion = nn.MSELoss()
        self.acc_func = HeatMapAcc()
        self.best_ap = 0.
        self.loss_logger = AverageLogger()
        self.acc_logger = AverageLogger()
        self.decoder = BasicKeyPointDecoder()

    def train(self, epoch):
        self.loss_logger.reset()
        self.acc_logger.reset()
        self.model.train()
        pbar = tqdm(self.tloader)
        print("#" * 25, "training start", "#" * 25)
        for i, (input_tensors, heat_maps, masks, _, _) in enumerate(pbar):
            input_img = input_tensors.to(self.inp_device)
            targets = heat_maps.to(self.out_device)
            mask = masks.to(self.out_device)
            self.optimizer.zero_grad()
            if self.scaler is None:
                predicts = self.model(input_img)
                loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                # self.lr_scheduler(self.optimizer, i, epoch)
                self.optimizer.step()
            else:
                with amp.autocast(enabled=True):
                    predicts = self.model(input_img)
                    loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]),
                                                targets.mul(mask[[..., None, None]]))
                self.scaler.scale(loss).backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                # self.lr_scheduler(self.optimizer, i, epoch)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            # self.ema.update(self.model)
            lr = self.optimizer.param_groups[0]['lr']
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]).detach(),
                                targets.mul(mask[[..., None, None]]).detach())
            self.loss_logger.update(loss.item())
            self.acc_logger.update(acc.item())
            pbar.set_description(
                "train epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}|lr:{:8.6f}".format(
                    epoch + 1,
                    i,
                    self.loss_logger.avg(),
                    self.acc_logger.avg() * 100,
                    lr,
                )
            )
        # self.ema.update_attr(self.model)
        self.lr_scheduler.step()
        print()
        print("#" * 25, "training end", "#" * 25)

    @torch.no_grad()
    def val(self, epoch):
        self.loss_logger.reset()
        self.acc_logger.reset()
        self.model.eval()
        # self.ema.ema.eval()
        pbar = tqdm(self.vloader)
        kps_dict_list = list()
        print("#" * 25, "evaluating start", "#" * 25)
        for i, (input_tensors, heat_maps, masks, trans_invs, img_ids) in enumerate(pbar):
            input_img = input_tensors.to(self.inp_device)
            targets = heat_maps.to(self.out_device)
            tran_inv = trans_invs.to(self.out_device)
            mask = masks.to(self.out_device)
            predicts = self.model(input_img)
            loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            self.loss_logger.update(loss.item())
            self.acc_logger.update(acc.item())
            pred_kps, scores = self.decoder(predicts, tran_inv)
            kps_to_dict_(pred_kps, scores, img_ids, kps_dict_list)
            pbar.set_description(
                "eval epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}".format(
                    epoch + 1,
                    i,
                    self.loss_logger.avg(),
                    self.acc_logger.avg() * 100,
                )
            )
        with open("temp_test.json", "w") as wf:
            json.dump(kps_dict_list, wf)
        val_ap = evaluate_map("temp_test.json", self.data_cfg['val_ann_path'])['AP']
        print("eval epoch:{:d}|mean_loss:{:8.6f}|mean_acc:{:6.4f}|val_ap:{:6.4f}".format(epoch + 1,
                                                                                         self.loss_logger.avg(),
                                                                                         self.acc_logger.avg() * 100,
                                                                                         val_ap))
        print("#" * 25, "evaluating end", "#" * 25)

        cpkt = {
            "ema": self.model.module.state_dict(),
            "epoch": epoch,
        }
        if val_ap > self.best_ap:
            self.best_ap = val_ap
            best_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_best.pth"
                                            .format(self.model_cfg['type']))
            torch.save(cpkt, best_weight_path)
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_last.pth"
                                        .format(self.model_cfg['type']))
        torch.save(cpkt, last_weight_path)

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
