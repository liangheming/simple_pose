import os
import json
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nets import pose_resnet_duc
from nets import pose_resnet_dconv
from torch.cuda import amp
from metrics.pose_metrics import HeatMapAcc, kps_to_dict_, BasicKeyPointDecoder, evaluate_map
from datasets.coco import MSCOCO
from commons.model_utils import rand_seed, ModelEMA, reduce_sum, AverageLogger
from commons.optims_utils import EpochWarmUpCosineDecayLRAdjust

rand_seed(512)


class DDPProcessor(object):
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
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus']
        self.gpus = str(self.cfg['gpus']).split(",")
        self.gup_num = len(self.gpus)
        dist.init_process_group(backend='nccl')
        self.tdata = MSCOCO(img_root=self.data_cfg['train_img_root'],
                            ann_path=self.data_cfg['train_ann_path'],
                            debug=self.data_cfg['debug'],
                            augment=True,
                            )
        train_sampler = DistributedSampler(dataset=self.tdata)
        batch_sampler_train = torch.utils.data.BatchSampler(
            train_sampler, self.data_cfg['batch_size'], drop_last=True)
        self.tloader = DataLoader(dataset=self.tdata,
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  batch_sampler=batch_sampler_train)
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
        )
        self.scaler = amp.GradScaler(enabled=True) if self.optim_cfg['amp'] else None
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.optim_cfg['lr']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.optim_cfg['milestones'],
            gamma=self.optim_cfg['gamma']
        )
        # self.lr_scheduler = EpochWarmUpCosineDecayLRAdjust(
        #     init_lr=self.optim_cfg['lr'],
        #     warm_up_epoch=0,
        #     epochs=self.optim_cfg['epochs'],
        #     iter_per_epoch=len(self.tloader)
        # )
        self.best_map = 0.
        local_rank = dist.get_rank()
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        model.to(self.device)
        if self.optim_cfg['sync_bn']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = nn.parallel.distributed.DistributedDataParallel(model,
                                                                     device_ids=[local_rank],
                                                                     output_device=local_rank)
        self.creterion = nn.MSELoss()
        self.acc_func = HeatMapAcc()
        self.loss_logger = AverageLogger()
        self.acc_logger = AverageLogger()
        self.decoder = BasicKeyPointDecoder()

    def train(self, epoch):
        self.model.train()
        self.loss_logger.reset()
        self.acc_logger.reset()
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
            print("#" * 25, "training start", "#" * 25)
        else:
            pbar = self.tloader

        for i, (input_tensors, heat_maps, masks, _, _) in enumerate(pbar):
            input_img = input_tensors.to(self.device)
            targets = heat_maps.to(self.device)
            mask = masks.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler is None:
                predicts = self.model(input_img)
                loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
                loss.backward()
                self.optimizer.step()
            else:
                with amp.autocast(enabled=True):
                    predicts = self.model(input_img)
                    loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]),
                                                targets.mul(mask[[..., None, None]]))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            lr = self.optimizer.param_groups[0]['lr']
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]).detach(),
                                targets.mul(mask[[..., None, None]]).detach())
            self.loss_logger.update(loss.item())
            self.acc_logger.update(acc.item())
            if self.local_rank == 0:
                pbar.set_description(
                    "train epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}|lr:{:8.6f}".format(
                        epoch + 1,
                        i,
                        self.loss_logger.avg(),
                        self.acc_logger.avg() * 100,
                        lr)
                )
        self.lr_scheduler.step()
        mean_loss = reduce_sum(torch.tensor(self.loss_logger.avg(), device=self.device)).item() / self.gup_num
        acc_mean = reduce_sum(torch.tensor(self.acc_logger.avg(), device=self.device)).item() / self.gup_num
        if self.local_rank == 0:
            print()
            print("train epoch:{:d}|mean_loss:{:8.6f}|mean_acc:{:6.4f}".format(epoch + 1,
                                                                               mean_loss,
                                                                               acc_mean * 100))
            print("#" * 25, "training end", "#" * 25)

    @torch.no_grad()
    def val(self, epoch):
        if self.local_rank != 0:
            return
        self.loss_logger.reset()
        self.acc_logger.reset()
        self.model.eval()
        pbar = tqdm(self.vloader)
        kps_dict_list = list()
        print("#" * 25, "evaluating start", "#" * 25)
        for i, (input_tensors, heat_maps, masks, trans_invs, img_ids) in enumerate(pbar):
            input_img = input_tensors.to(self.device)
            targets = heat_maps.to(self.device)
            tran_inv = trans_invs.to(self.device)
            mask = masks.to(self.device)
            predicts = self.model.module(input_img)
            loss = 0.5 * self.creterion(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            acc = self.acc_func(predicts.mul(mask[[..., None, None]]), targets.mul(mask[[..., None, None]]))
            pred_kps, scores = self.decoder(predicts, tran_inv)
            kps_to_dict_(pred_kps, scores, img_ids, kps_dict_list)
            self.loss_logger.update(loss.item())
            self.acc_logger.update(acc.item())
            pbar.set_description(
                "val epoch:{:3d}|iter:{:4d}|loss:{:8.6f}|acc:{:6.4f}".format(
                    epoch + 1,
                    i,
                    self.loss_logger.avg(),
                    self.acc_logger.avg() * 100,
                )
            )
        cpkt = {
            "ema": self.model.module.state_dict(),
            "epoch": epoch,
        }
        with open("temp_test.json", "w") as wf:
            json.dump(kps_dict_list, wf)
        val_ap = evaluate_map("temp_test.json", self.data_cfg['val_ann_path'])['AP']

        print("val epoch:{:d}|mean_loss:{:8.6f}|mean_acc:{:6.4f}|val_ap:{:6.4f}".format(epoch + 1,
                                                                                        self.loss_logger.avg(),
                                                                                        self.acc_logger.avg() * 100,
                                                                                        val_ap))
        print("#" * 25, "evaluating end", "#" * 25)
        if val_ap > self.best_map:
            self.best_map = val_ap
            best_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_best.pth"
                                            .format(self.cfg['model_name']))
            torch.save(cpkt, best_weight_path)

        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_last.pth"
                                        .format(self.cfg['model_name']))
        torch.save(cpkt, last_weight_path)

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
