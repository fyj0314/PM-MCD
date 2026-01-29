import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.lovasz_loss as L

from tqdm import tqdm
from models.PMMCD import PMMCD
from configs.config import get_config
from torch.utils.data import DataLoader
from datasets.LandsatSCD.make_data_loader import Datset
from utils.metric_landsat import accuracy, SCDD_eval_all, AverageMeter


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss > self.best_val_loss - self.min_delta:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        train_data = Datset(args.train_dataset_path, args.train_data_name_list, None, None, 'train')
        val_data = Datset(args.val_dataset_path, args.val_data_name_list, None, None, 'val')
        self.train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.deep_model = PMMCD(
            Cconc=256,
            n_class=10,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        # self.deep_model = SVMamba(Cconc=256, n_class=10)
        self.deep_model = self.deep_model.cuda()

        self.epoch = args.epoch

        self.optimizer = torch.optim.AdamW(self.deep_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

        self.best_miou = 0.0

    # train loop
    def training(self):
        for epoch in range(self.epoch):
            time_start = time.time()
            train_num = 0
            total_train_loss = 0.0
            torch.cuda.empty_cache()
            self.deep_model.train()

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                img1, img2, label, _ = batch
                img1, img2, label = img1.cuda(), img2.cuda(), label.cuda().long()

                output = self.deep_model(img1, img2)  # forward

                self.optimizer.zero_grad()

                ce_loss_clf = F.cross_entropy(output, label)
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output, dim=1), label)
                loss = ce_loss_clf + 0.75 * lovasz_loss_clf

                total_train_loss += loss
                train_num += 1

                loss.backward()
                self.optimizer.step()

            Pre, Rec, Ious, F1, OA, val_loss = self.validation()
            IoU_mean = np.mean(Ious)

            train_loss = total_train_loss / train_num

            elapsed = round(time.time() - time_start)

            with open('result1/metric.txt', 'a') as file:
                file.write(f'{epoch}：Pre is {np.mean(Pre)}, Rec is {np.mean(Rec)}, F1 is {np.mean(F1)}, OA is {OA}, mIoU is {IoU_mean}, Elapsed is {elapsed}\n')
            with open('result1/train_loss.txt', 'a') as file:
                file.write(f"{epoch}：{train_loss}\n")
            with open('result1/val_loss.txt', 'a') as file:
                file.write(f"{epoch}：{val_loss}\n")

            if self.best_miou < IoU_mean:
                self.best_miou = IoU_mean
                state = {'net': self.deep_model.state_dict()}
                torch.save(state, os.path.join('result1', 'model_' + str(epoch) + '.pth'))

            print('Epoch {}, train_loss {}, val_loss {}, Elapsed {}.'.format(epoch, train_loss, val_loss, elapsed))
            print(f'{epoch}：Pre is {np.mean(Pre)}, Rec is {np.mean(Rec)}, F1 is {np.mean(F1)}, OA is {OA}, mIoU is {IoU_mean}')

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch}')
                break

    def validation(self):
        self.deep_model.eval()

        val_num = 0
        total_val_loss = 0.0
        torch.cuda.empty_cache()
        acc_meter = AverageMeter()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_loader)):
                img1, img2, label, _ = batch
                img1, img2, label = img1.cuda(), img2.cuda(), label.cuda().long()

                output = self.deep_model(img1, img2)

                ce_loss_clf = F.cross_entropy(output, label)
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output, dim=1), label)
                loss = ce_loss_clf + 0.75 * lovasz_loss_clf

                total_val_loss += loss
                val_num += 1

                for (pred_scd, label_scd) in zip(torch.argmax(output, dim=1).cpu(), label.cpu()):
                    acc_A, valid_sum_A = accuracy(pred_scd, label_scd)
                    preds_all.append(pred_scd)
                    labels_all.append(label_scd)
                    acc = acc_A
                    acc_meter.update(acc)

        Pre, Rec, Ious, F1 = SCDD_eval_all(preds_all, labels_all, 10)
        val_loss = total_val_loss / val_num

        return Pre, Rec, Ious, F1, acc_meter.avg, val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vssm_svmamba.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_dataset_path', type=str, default='Landsat811/train')
    parser.add_argument('--train_data_list_path', type=str, default='datasets/LandsatSCD/train_list811.txt')
    parser.add_argument('--val_dataset_path', type=str, default='Landsat811/val')
    parser.add_argument('--val_data_list_path', type=str, default='datasets/LandsatSCD/val_list811.txt')
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [train_data_name.strip() for train_data_name in f]
    args.train_data_name_list = train_data_name_list
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [val_data_name.strip() for val_data_name in f]
    args.val_data_name_list = val_data_name_list

    trainer = Trainer(args)
    trainer.training()