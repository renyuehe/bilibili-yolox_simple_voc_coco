#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from functools import partial


class LRScheduler:
    def __init__(self, name, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        学习率策略类
        Supported lr schedulers: [cos, warmcos, multistep]

        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - cos: None
                - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
                - multistep: [milestones (epochs), gamma (default 0.1)]
        """

        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)

        self.lr_func = self._get_lr_func(name)

    # 更新学习率
    def update_lr(self, iters):
        return self.lr_func(iters)

    # 通过 name 的方式来更新学习率
    def _get_lr_func(self, name):
        if name == "cos":  # cosine lr schedule
            lr_func = partial(cos_lr, self.lr, self.total_iters)
        elif name == "warmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
            lr_func = partial(
                warm_cos_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        elif name == "yoloxwarmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
            lr_func = partial(
                yolox_warm_cos_lr,
                self.lr,
                min_lr_ratio,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
                no_aug_iters,
            )
        elif name == "yoloxsemiwarmcos":
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            normal_iters = self.iters_per_epoch * self.semi_epoch
            semi_iters = self.iters_per_epoch_semi * (
                self.total_epochs - self.semi_epoch - self.no_aug_epochs
            )
            lr_func = partial(
                yolox_semi_warm_cos_lr,
                self.lr,
                min_lr_ratio,
                warmup_lr_start,
                self.total_iters,
                normal_iters,
                no_aug_iters,
                warmup_total_iters,
                semi_iters,
                self.iters_per_epoch,
                self.iters_per_epoch_semi,
            )
        elif name == "multistep":  # stepwise lr schedule
            milestones = [
                int(self.total_iters * milestone / self.total_epochs)
                for milestone in self.milestones
            ]
            gamma = getattr(self, "gamma", 0.1)
            lr_func = partial(multistep_lr, self.lr, milestones, gamma)
        else:
            raise ValueError("Scheduler version {} not supported.".format(name))
        return lr_func

'''
余弦算法
total_iters: 半个周期
iters: x 轴坐标点
lr: 学习率的范围是 0 - lr
'''
def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr

'''
余弦退火算法

在 warmup_total_iters 之前是 线性策略： 从 warmup_lr_start 至 lr
在 warmup_total_iters 之后是 余弦策略:  就是 cos_lr
'''
def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr

'''
第一个阶段：从 warmup_lr_start 开始, 通过 warmup_total_iters 个步数 , 变成 lr 的值
第二个阶段：从 lr 开始，执行 total_iters - no_aug_iter 个步数 , 变成 lr * min_lr_ratio 的值
第三个阶段：lr = lr * min_lr_ratio

补充：实际上 total_iters 和 no_aug_iter 都是同时使用的,而且都是 total_iters - no_aug_iter, 所以这两个变量可以看成是一个变量
'''
def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr


'''
类比于上一个函数
第一阶段：iters <= warmup_total_iters阶段, 起始值 warmup_lr_start, 经过 warmup_total_iters 个步数， 数值到达 lr。
第二个阶段：iters < normal_iters 阶段， total_iters（周期）、lr最高、lr * min_lr_ratio最低
第三个阶段： iters > normal_iters 阶段， lr * min_lr_radio
'''
def yolox_semi_warm_cos_lr(
    lr,
    min_lr_ratio,
    warmup_lr_start,
    total_iters,
    normal_iters,
    no_aug_iters,
    warmup_total_iters,
    semi_iters,
    iters_per_epoch,
    iters_per_epoch_semi,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= normal_iters + semi_iters:
        lr = min_lr
    elif iters <= normal_iters:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (
                    normal_iters
                    - warmup_total_iters
                    + (iters - normal_iters)
                    * iters_per_epoch
                    * 1.0
                    / iters_per_epoch_semi
                )
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    return lr

'''
步长算法
iters < milestone 时为 1
iters > milestone 时为 逐步 lr *= gamma
'''
def multistep_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate"""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr
