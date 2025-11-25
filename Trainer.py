# -*- encoding: utf-8 -*-
"""
@type: module

@brief: Trainer for Data Fusion Task.

@author: guanshikang

Created on Mon Oct 20 14:53:53 2025, HONG KONG
"""
import os
import re
import time
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from metrics import *
from sklearn.model_selection import KFold
from TrainDataLoader import TrainDataLoader
from FunctionalCode.GlobalStats import ComputeStats
from FunctionalCode.StatsPlotFuncs import StatsPlot
from Backbone.SwinTransformer_v3 import SwinTransformer

class Trainer:
    def __init__(self,
                 cfg,  # Configuration file dict, required.
                 checkpoint_path=None,  # referred checkpoint path.
                 model=None,  # custom model. If None, load SwinTransformer.
                 optimizer=None,  # custom optimizer. If None, default to AdamW.
                 scheduler=None,  # custom scheduler. If None, default to LinearLR warmup and CosineAnnealingLR.
                 criterion=None  # custom criterion. If None, default to SmoothL1Loss
                ):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.SmoothL1Loss() if criterion is None else criterion
        self.device = torch.device(self.cfg['base']['device'] if torch.cuda.is_available() else "cpu")

        # Training state
        self.start_epoch = 1
        self.best_r2 = -np.inf
        self.best_psnr = -np.inf
        self.train_psnr = -np.inf
        self.last_epoch_r2 = 0.0
        self.best_epoch = 0
        self.metrics_dict = self.initialize_metrics_dict()

        # DataLoaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def build(self):
        """Build all components of Trainer."""
        self.prepare_dataloader()
        if self.model is None:
            self.setup_model()
        if self.optimizer is None or self.scheduler is None:
            self.setup_optimizer_scheduler()
        if self.cfg['status']['checkpoint']:
            self.load_checkpoint()
            self.start_epoch = self.best_epoch + 1

    def run_train(self):
        for epoch in range(self.start_epoch, self.cfg['base']['epochs'] + 1):
            stime = time.time()
            self.train_dataset.update_progressive_epoch(epoch)
            train_metrics = self.train(epoch)
            self.update_metrics(train_metrics, 'train')
            print("Epoch: %d - Train: R2: %g, RMSE: %g, PSNR: %g, SSIM: %g" %
                  (epoch, train_metrics['r2'], train_metrics['rmse'],
                   train_metrics['psnr'], train_metrics['ssim']))
            etime = time.time()
            print(f"Training epoch {epoch} used {etime - stime:.4f} s.")
            if epoch % 2 == 0:
                stime = time.time()
                self.val_dataset.update_progressive_epoch(epoch)
                val_metrics, pred_list, label_list = self.validate(epoch)
                self.update_metrics(val_metrics, 'val')
                print("Epoch: %d - Validation: R2: %g, RMSE: %g, PSNR: %g, SSIM: %g" %
                      (epoch, val_metrics['r2'], val_metrics['rmse'],
                       val_metrics['psnr'], val_metrics['ssim']))
                if (val_metrics['r2'] > self.best_r2) and (val_metrics['psnr'] > self.best_psnr):
                    self.best_r2 = val_metrics['r2']
                    self.best_psnr = val_metrics['psnr']
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, pred_list, label_list, val_metrics)
                    print("Find best model in epoch %d with R2 score of %f and PSNR of %f" %
                          (epoch, self.best_r2, self.best_psnr))
                    print("SAM: %.4f, ERGAS: %.4f" % (val_metrics['sam'], val_metrics['ergas']))
                    band_metrics = val_metrics.get('band_metrics', {})
                    for band_name, metrics in band_metrics.items():
                        print(
                            "%s, R2: %.4f, RMSE: %.4f, PSNR: %.4f, SSIM: %.4f" %
                            (band_name, metrics['r2'], metrics['rmse'], metrics['psnr'], metrics['ssim'])
                        )
                etime = time.time()
                print(f"Evaluation epoch {epoch} used {etime - stime:.4f} s.")

        self.save_metrics()

    def run_test(self):
        print("\n" + "-" * 25 + " TEST STAGE " + "-" * 25)
        if os.path.exists(self.checkpoint_path):
            check_point = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(check_point['model'])
            self.val_dataset.update_progressive_epoch(check_point['best_epoch'])
            print(f"\nLoaded checkpoint from {self.checkpoint_path}")
        else:
            print("\nNo checkpoint provided, using current model weights.")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        self.test()

    def prepare_dataloader(self):
        df = pd.read_csv(self.cfg['base']['dataset_file'])
        train_files = df['train'].dropna().to_list()
        val_files = df['val'].dropna().to_list()
        test_files = None
        if 'test' in df.keys():
            test_files = df['test'].dropna().to_list()
        if not self.cfg['status']['index_ahead']:  # random select 1 fold to split original dataset
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            temp_files = train_files + val_files
            train_idx, val_idx = kf.split(temp_files).__next__()
            train_files = [temp_files[i] for i in train_idx]
            val_files = [temp_files[i] for i in val_idx]
            stime = time.time()
            train_files, val_files, test_files = self.pre_indexing([train_files,
                                                                    val_files,
                                                                    test_files])
            etime = time.time()
            print(f"Searching index used {etime - stime:.4f} s.")
        stime = time.time()
        self.compute_stats(train_files)
        etime = time.time()
        print(f"Computing stats used {etime - stime:.4f} s.\n")

        dataset, dataloader = [], []
        for files, shuffle, stage in zip([train_files, val_files, test_files],
                                         [True, False, False],
                                         ["train", "val", "test"]):
            if files is None:
                temp_dataset = None
                temp_loader = None
            else:
                temp_dataset, temp_loader = self.create_dataloader(
                    files,
                    stage,
                    use_custom_fn=True,
                    shuffle=shuffle,
                    pin_memory=True,
                    num_workers=min(6, os.cpu_count()),
                    persistent_workers=True,
                    prefetch_factor=4
                )
            dataset.append(temp_dataset)
            dataloader.append(temp_loader)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader

    def pre_indexing (self, file_ls):
        """iter validation and test dataset for their indexing."""
        new_files = []
        df = pd.DataFrame({})
        for files, stage in zip(file_ls, ["train", "val", "test"]):
            if files is None:
                temp_loader = None
                new_files.append(None)
            else:
                print(f"Searching {stage} dataset index ...")
                _, temp_loader = self.create_dataloader(files,
                                                        stage,
                                                        indexing=True,
                                                        use_custom_fn=False,
                                                        batch_size=24,
                                                        shuffle=False,
                                                        pin_memory=False,
                                                        num_workers=min(6, os.cpu_count()),
                                                        persistent_workers=False,
                                                        prefetch_factor=4)
                temp_ls = []
                for batch in temp_loader:
                    label_path, landsat_idx, modix_idx = batch
                    for path, lidx, midx in zip(label_path, landsat_idx, modix_idx):
                        temp_dict = {
                            'label_path': path,
                            'landsat_idx': lidx.numpy().tolist(),
                            'modis_idx': midx.numpy().tolist()
                        }
                        temp_ls.append(temp_dict)
                new_files.append(temp_ls)
                df = pd.concat([df, pd.DataFrame({stage: temp_ls})], axis=1)
        save_path = self.cfg['base']['dataset_file'].replace(".csv", "_new.csv")
        df.to_csv(save_path)
        print(f"Index searching result is saved in {save_path} at {datetime.now()}")

        return new_files

    def compute_stats(self, files):
        if not self.cfg['status']['stats_ahead']:
            print("\nCalculating statistical indicators ...")
            self.cfg['dataloader']['stats'] = None
            cgs = ComputeStats()
            _, stats_dataloader = self.create_dataloader(files,
                                                         "train",
                                                         use_custom_fn=False,
                                                         batch_size=24,
                                                         shuffle=False,
                                                         pin_memory=False,
                                                         num_workers=min(6, os.cpu_count()),
                                                         persistent_workers=True,
                                                         prefetch_factor=4)
            stats = {}
            for flag, band_num in zip(["landsat", "modis_Q", "modis_A"], [6, 2, 9]):
                mean, std = cgs.compute_global_stats(
                    stats_dataloader, flag, mode='std', category=self.cfg['base']['category'],
                    save_dir=self.cfg['base']['save_dir'], channel_num=band_num
                )
                stats[f"{flag}_mean"] = mean
                stats[f"{flag}_std"] = std
            self.cfg['dataloader']['stats'] = stats
        else:
            stats = self.cfg['dataloader']['stats']
        print("\nStatistical Indicators:")
        print("-" * 50)
        print("Landsat:\nMean: {0}\nStd: {1}\n".format(stats['landsat_mean'], stats['landsat_std']))
        print("MODIS_Q:\nMean: {0}\nStd: {1}\n".format(stats['modis_Q_mean'], stats['modis_Q_std']))
        print("MODIS_A:\nMean: {0}\nStd: {1}".format(stats['modis_A_mean'], stats['modis_A_std']))
        print("-" * 50)

    def create_dataloader(self, files, stage, indexing=False, **kwargs):
        if files is None or len(files) == 0:
            print("Provided labels are invalid, please check.")
            os._exit(1)
        self.cfg['dataloader']['files'] = files
        dataset_name = os.path.split(self.cfg['base']['dataset_file'])[1].split(".")[0]
        update_dataset_name = f"{dataset_name}_{stage}"
        dataset = TrainDataLoader(indexing=indexing,
                                  update_dataset_file=update_dataset_name,
                                  **self.cfg['dataloader'])

        custom_fn = None
        use_custom_fn = kwargs.get('use_custom_fn', False)
        if use_custom_fn:
            custom_fn = lambda batch: (
                torch.stack([x['landsat'] for x in batch]),
                torch.stack([x['modis_Q'] for x in batch]),
                torch.stack([x['modis_A'] for x in batch]),
                torch.stack([x['label'] for x in batch]),
                torch.stack([x['gt_mask'] for x in batch])
            )
        batch_size = kwargs.get('batch_size', self.cfg['base']['batch_size'])
        shuffle = kwargs.get('shuffle', True)
        pin_memory = kwargs.get('pin_memory', False)
        num_workers = kwargs.get('num_workers', 0)
        persistent_workers = False if num_workers == 0 else kwargs.get('persistent_workers', False)
        prefetch_factor = kwargs.get('prefetch_factor', 2)

        dataloder = data_utils.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          collate_fn=custom_fn,
                                          pin_memory=pin_memory,
                                          num_workers=num_workers,
                                          persistent_workers=persistent_workers,
                                          prefetch_factor=prefetch_factor)

        return dataset, dataloder

    def setup_model(self):
        """Initialize model and move to device"""
        print("Initialize model from default SwinTransformer.")
        self.model = SwinTransformer(**self.cfg['model_params']).to(self.device)

    def setup_optimizer_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        if self.optimizer is None:
            print("Initialize optimizer from AdamW.")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg['base']['lr'],
                weight_decay=5e-4
            )

        if self.scheduler is None:
            print("Initialize scheduler from LinearLR warmup and CosineAnnealingLR.\n")
            total_train_batches = len(self.train_dataloader)
            accumulation_steps = self.cfg['base']['accumulation_steps']
            updates_per_epoch = (total_train_batches + accumulation_steps - 1) // accumulation_steps

            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(
                        self.optimizer,
                        start_factor=0.01,
                        end_factor=1,
                        total_iters=self.cfg['base']['warm_up'] * updates_per_epoch
                    ),
                    CosineAnnealingLR(
                        self.optimizer,
                        T_max=(self.cfg['base']['epochs'] - self.cfg['base']['warm_up']) * updates_per_epoch,
                        eta_min=5e-6
                    )
                ],
                milestones=[self.cfg['base']['warm_up'] * updates_per_epoch]
            )

    def train(self, epoch):
        self.model.train()
        pred_list, label_list = [], []
        count = 0
        eval_dict = {
            'sum_y': 0., 'sum_y2': 0., 'sum_pred': 0., 'sum_pred2': 0.,
            'sum_y_pred': 0., 'n_pixels': 0, 'ssim_list': [],
            'psnr_list': [], 'sam_list': [], 'ergas_list': [],
        }
        loss_metric = 0
        self.optimizer.zero_grad()
        for iter, batch in enumerate(self.train_dataloader):
            logits, label, gt_mask = self.get_logits(batch)
            loss, all_band_ssim = self.criterion(logits * gt_mask, label * gt_mask)
            scaled_loss = loss / self.cfg['base']['accumulation_steps']
            scaled_loss.backward()
            loss_metric += loss.item()
            count += 1
            if ((iter + 1) % self.cfg['base']['accumulation_steps'] == 0) or \
                ((iter + 1) == len(self.train_dataloader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            eval_dict = batch_evaluation(logits, label, eval_dict, all_band_ssim)
            if iter % 10 == 0:
                if len(pred_list) <= 10:
                    pred_list.append(logits.cpu().detach().numpy())
                    label_list.append(label.cpu().detach().numpy())
                current_avg_loss = loss_metric / count
                print("epoch: %d, iter: %d, lr: %g, loss: %g" %
                      (epoch, iter, self.optimizer.param_groups[0]['lr'], current_avg_loss))

            del logits, label
            torch.cuda.empty_cache()
        r2, rmse, psnr, ssim, sam, ergas = epoch_evaluation(eval_dict)
        if psnr > self.train_psnr:
            self.train_psnr = psnr
            save_path = os.path.join(self.cfg['base']['save_dir'], "train_files",
                                    f"train_result_{self.cfg['base']['category']}.npz")
            pred_list = np.concatenate(pred_list, dtype=np.float32)
            label_list = np.concatenate(label_list, dtype=np.float32)
            np.savez(save_path, pred=pred_list, label=label_list)

        return {
            'loss': loss_metric / count,
            'r2': r2,
            'rmse': rmse,
            'psnr': psnr,
            'ssim': ssim,
            'sam': sam,
            'ergas': ergas
        }

    def validate(self, epoch):
        self.model.eval()
        label_list, pred_list = [], []
        count = 0
        eval_dict = {
            'sum_y': 0., 'sum_y2': 0., 'sum_pred': 0., 'sum_pred2': 0.,
            'sum_y_pred': 0., 'n_pixels': 0, 'ssim_list': [],
            'psnr_list': [], 'sam_list':[], 'ergas_list': []
        }
        loss_metric = 0
        compute_band = self.compute_band_metrics(epoch)
        with torch.no_grad():
            for batch in self.val_dataloader:
                logits, label, gt_mask = self.get_logits(batch)
                loss, all_band_ssim = self.criterion(logits * gt_mask, label * gt_mask)

                loss_metric += loss.item()
                count += 1

                eval_dict = batch_evaluation(logits, label, eval_dict,
                                             all_band_ssim,
                                             compute_band=compute_band)
                if len(pred_list) <= 30:
                    pred_list.append(logits.cpu().detach().numpy())
                    label_list.append(label.cpu().detach().numpy())

                del logits, label
                torch.cuda.empty_cache()
        r2, rmse, psnr, ssim, sam, ergas = epoch_evaluation(eval_dict)
        self.last_epoch_r2 = r2
        band_metrics = {}
        if compute_band and 'band_stats' in eval_dict:
            band_metrics = band_evaluation(eval_dict['band_stats'],
                                           self.cfg['base']['band_names'])
        return{
            'loss': loss_metric / count,
            'rmse': rmse,
            'r2': r2,
            'psnr': psnr,
            'ssim': ssim,
            'sam': sam,
            'ergas': ergas,
            'band_metrics': band_metrics,
        }, pred_list, label_list

    def test(self):
        self.model.eval()
        label_list, pred_list = [], []
        eval_dict = {
            'sum_y': 0., 'sum_y2': 0., 'sum_pred': 0., 'sum_pred2': 0.,
            'sum_y_pred': 0., 'n_pixels': 0, 'ssim_list': [],
            'psnr_list': [], 'sam_list': [], 'ergas_list': []
        }
        count = 0
        loss_metric = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                logits, label, gt_mask = self.get_logits(batch)
                loss, all_band_ssim = self.criterion(logits * gt_mask, label * gt_mask)

                loss_metric += loss.item()
                count += 1

                eval_dict = batch_evaluation(logits, label, eval_dict, all_band_ssim)
                pred_list.append(logits.cpu().detach().numpy())
                label_list.append(label.cpu().detach().numpy())

                del logits, label
                torch.cuda.empty_cache()

        r2, rmse, psnr, ssim, sam, ergas = epoch_evaluation(eval_dict)
        print("\nTEST RESULTS SUMMARY")
        print("-" * 50)
        print("Test Loss:  %.6f." % (loss_metric / count))
        print("Test R2:    %.6f." % r2)
        print("Test RMSE:  %.6f." % rmse)
        print("Test PSNR:  %.6f." % psnr)
        print("Test SSIM:  %.6f." % ssim)
        print("Test SAM:   %.6f." % sam)
        print("Test ERGAS: %.6f." % ergas)
        print("-" * 50 + "\n")
        label_list = np.concatenate(label_list, dtype=np.float32)
        pred_list = np.concatenate(pred_list, dtype=np.float32)
        save_path = os.path.join(
            self.cfg['base']['save_dir'], "test_files",
            f"test_result_{self.cfg['base']['category']}.npz"
        )
        np.savez(save_path, label=label_list, pred=pred_list)


    def save_checkpoint(self, epoch, pred_list, label_list, val_metrics):
            model_state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_epoch': epoch,
            }
            model_state.update(val_metrics)
            save_path = os.path.join(
                self.cfg['base']['save_dir'], "checkpoint",
                f"checkpoint_{self.cfg['base']['category']}_ep{epoch}.pth"
            )
            self.checkpoint_path = save_path  # update checkpoint path in training for test stage
            torch.save(model_state, save_path)
            label_list = np.concatenate(label_list, dtype=np.float32)
            pred_list = np.concatenate(pred_list, dtype=np.float32)
            save_path = os.path.join(
                self.cfg['base']['save_dir'], "val_files",
                f"val_result_{self.cfg['base']['category']}.npz"
            )
            np.savez(save_path, label=label_list, pred=pred_list)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("There is no referred checkpoint, please check!")
            print("Searching latest version from save directory...")
            checkpoint_dir = os.path.join(self.cfg['base']['save_dir'], "checkpoint")
            pattern = r"^checkpoint_{}_ep\d+.pth$".format(self.cfg['base']['category'])
            checkpoint_names = list(filter(lambda x: re.match(pattern, x), os.listdir(checkpoint_dir)))
            if len(checkpoint_names) > 0:
                checkpoint_names.sort()
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[-1])
                self.checkpoint_path = checkpoint_path
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"No results, training from inital state...")
                self.best_epoch = 0
                self.best_r2 = -np.inf
                self.best_psnr = -np.inf

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            if self.optimizer and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.best_r2 = checkpoint.get('best_r2', -np.inf)
            self.best_psnr = checkpoint.get('best_psnr', -np.inf)

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from initial state...")
            self.best_epoch = 0
            self.best_r2 = -np.inf
            self.best_psnr = -np.inf

    def initialize_metrics_dict(self):
        metrics_dict = {}
        for stage in ['train', 'val']:
            for metric in ['loss', 'r2', 'rmse', 'psnr', 'ssim', 'sam', 'ergas']:
                metrics_dict[f'{stage}_{metric}'] = []
            if stage == "val":
                metrics_dict[f'{stage}_band_metrics'] = []

        return metrics_dict

    def save_metrics(self):
        save_path = os.path.join(self.cfg['base']['save_dir'], "metrics",
                                f"metrics_{self.cfg['base']['category']}.pkl")
        with open(save_path, 'wb') as f:
            pkl.dump(self.metrics_dict, f)
        key = ["loss", "r2", "rmse", "psnr", "ssim", "sam", "ergas"]
        sp = StatsPlot()
        file_name = f"./loss_pics/metric_plot_{self.cfg['base']['category']}.png"
        sp.line_plot(self.metrics_dict, key, col=4, file_name=file_name)

    def update_metrics(self, metrics, stage):
        for k, v in metrics.items():
            self.metrics_dict[f'{stage}_{k}'].append(v)

    def get_logits(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        landsat, modisQ, modisA, label, gt_mask = batch
        landsat = landsat.to(self.device, non_blocking=True)
        modisQ = modisQ.to(self.device, non_blocking=True)
        modisA = modisA.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        gt_mask = gt_mask.to(self.device, non_blocking=True)
        logits = self.model(landsat, modisQ, modisA)

        del landsat, modisQ, modisA
        return logits, label, gt_mask

    def compute_band_metrics(self, current_epoch) -> bool:
        """
        Combine multiple conditions to ensure band accuray calculated.
        """
        total_epochs = self.cfg['base']['epochs']
        # every 20 epochs at early stage.
        if current_epoch <= total_epochs * 0.3 and current_epoch % 20 == 0:
            return True

        # last epoch r2 less than 95% of best r2. if training is stable, it should work.
        if total_epochs * 0.3 < current_epoch <= total_epochs * 0.6 and self.last_epoch_r2 > self.best_r2 * 0.95:
            return True

        # more than 10 epochs didn't calculate (5 evaluations).
        if current_epoch - self.best_epoch >= 10:
            return True

        # every 8 epochs at middle stage (4 evalutions.).
        if total_epochs * 0.6 < current_epoch <= total_epochs * 0.8 and current_epoch % 8 == 0:
            return True

        # every 4 epochs at late stage (2 evalutions.).
        if total_epochs * 0.8 < current_epoch <= total_epochs and current_epoch % 4 == 0:
            return True

        return False
