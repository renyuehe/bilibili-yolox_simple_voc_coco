#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher

from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # init函数只定义一些基本的attr，其他atr，如模型，优化器内置

        # before_train methods.
        self.exp = exp     #网络
        self.args = args   #参数

        # training related attr
        self.max_epoch = exp.max_epoch # 最大批次

        self.scaler = torch.cuda.amp.GradScaler(enabled=False) # 通过精度求得scaler目的是为了解决在混合精度运算的时候半精度带来的underflow

        self.device = "cuda:{}".format(0) # gpu编号

        self.use_model_ema = exp.ema # bool值:默认位True

        # data/dataloader related attr
        self.input_size = exp.input_size # 尺寸(640,640)
        self.best_ap = 0 #

        # metric record
        # 十进制 记录
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name) # YOLOX_outputs/yolox_voc_s

        os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            filename="train_log.txt",
            mode="a",
        )

    #! -------------- train层 --------------- !#
    # 训练 中
    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    # 训练 前
    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(0)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs

        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size
        )

        # Tensorboard logger
        self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    # 训练 后
    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )



    # ! -------------- epoch 层 --------------- !#
    # 一个 train == start epoch 至 max epoch 中
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    # 一个 epoch 前
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")

            self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
    # 一个 epoch 后
    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()



    # ! -------------- iter 层 --------------- !#
    # 一个epoch == 多个iter
    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    # 一个 iter 前
    def before_iter(self):
        pass

    # 一个 iter 后
    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch
            )


    # ! -------------- iter 内层 --------------- !#
    # 一个 iter
    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(torch.float32)
        targets = targets.to(torch.float32)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        outputs = self.model(inps, targets)
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )


    # 当前进度
    @property
    def progress_in_iter(self):
        # 一个train 有多个 epoch, 一个 epoch 有多个 iter
        return self.epoch * self.max_iter + self.iter

    # 恢复训练
    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    # 评估保存的模型
    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator
        )
        self.model.train()

        self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
        self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
        logger.info("\n" + summary)

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    # 保存到 ckpt
    def save_ckpt(self, ckpt_name, update_best_ckpt=False):

        save_model = self.ema_model.ema if self.use_model_ema else self.model

        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )
