import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from nets.fastpose import FastPose
from metrics.pose_metrics import HeatMapAcc
from datasets.coco import MSCOCO
from commons.model_utils import rand_seed, ModelEMA
from torch.optim.adam import Adam

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
        model = FastPose(
            num_joints=self.model_cfg['num_joints'],
            backbone=self.model_cfg['backbone'],
            reduction=self.model_cfg['reduction']
        )
        self.optimizer = Adam(
            model.parameters(), lr=self.optim_cfg['lr']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.optim_cfg['milestones'],
            gamma=self.optim_cfg['gamma']
        )
        assert torch.cuda.is_available(), "training only support cuda"
        assert torch.cuda.device_count() >= len(self.cfg['gpus']), "not have enough gpus"
        self.inp_device = torch.device("cuda:{:d}".format(self.cfg['gpus'][0]))
        self.out_device = torch.device("cuda:{:d}".format(self.cfg['gpus'][-1]))
        model.to(self.inp_device)
        # if self.optim_cfg['sync_bn']:
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = nn.DataParallel(
            model, device_ids=self.cfg['gpus'], output_device=self.out_device)

        self.ema = ModelEMA(self.model)
        self.creterion = nn.MSELoss()
        self.acc_func = HeatMapAcc()

    def train(self, epoch):
        self.model.train()
        loss_list = list()
        acc_list = list()
        pbar = tqdm(self.tloader)
        lr = 0
        for i, input_info in enumerate(pbar):
            input_img = input_info['input'].to(self.inp_device)
            targets = input_info['targets'].to(self.out_device)
            mask = input_info['mask'].to(self.out_device)
            predicts = self.model(input_img)
            loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.model)
            lr = self.optimizer.param_groups[0]['lr']
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]).detach(),
                                targets.mul(mask[[..., None, None]]).detach())
            loss_list.append(loss.item())
            acc_list.append(acc.item())
            pbar.set_description(
                "epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}|lr:{:8.6f}|mask:{:4d}".format(
                    epoch + 1,
                    i,
                    loss.item(),
                    acc.item() * 100,
                    lr,
                    int(mask.sum().item()))
            )
        self.ema.update_attr(self.model)
        self.lr_scheduler.step()
        mean_loss = np.array(loss_list).mean()
        acc_mean = np.array(acc_list).mean()
        print("train epoch:{:d}|mean_loss:{:8.6f}|mean_acc:{:6.4f}|lr:{:8.6f}".format(epoch + 1,
                                                                                      mean_loss,
                                                                                      acc_mean * 100,
                                                                                      lr))

    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        self.ema.ema.eval()
        loss_list = list()
        acc_list = list()
        pbar = tqdm(self.vloader)
        for i, input_info in enumerate(pbar):
            input_img = input_info['input'].to(self.inp_device)
            targets = input_info['targets'].to(self.inp_device)
            mask = input_info['mask'].to(self.inp_device)
            predicts = self.ema.ema(input_img)
            loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            pbar.set_description(
                "epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}".format(
                    epoch + 1,
                    i,
                    loss.item(),
                    acc.item() * 100,
                )
            )
            loss_list.append(loss.item())
            acc_list.append(acc.item())
        mean_loss = np.array(loss_list).mean()
        acc_mean = np.array(acc_list).mean()
        print("val epoch:{:d}|mean_loss:{:6.4f}|mean_acc:{:6.4f}".format(epoch + 1,
                                                                         mean_loss,
                                                                         acc_mean * 100))
        cpkt = {
            "ema": self.ema.ema.state_dict(),
            "epoch": epoch,
        }
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_last.pth"
                                        .format(self.cfg['model_name']))
        torch.save(cpkt, last_weight_path)

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
