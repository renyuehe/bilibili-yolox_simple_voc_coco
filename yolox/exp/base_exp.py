#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module

from yolox.utils import LRScheduler # 学习率策略


class BaseExp(metaclass=ABCMeta):
    """
    BaseExp类，
    一、组合了 模型、dataloader、优化器、学习率策略器
    二、通过读取配置文件的方式来设置类的属性

    brief: Basic class for any experiment.
    get_model: 获取模型
    get_data_loader: 获取数据集
    get_optimizer: 获取优化器
    get_lr_scheduler: 获取学习率策略器
    get_evaluator: ？？？ 获取评估
    eval: ？？？ 获取指定 model 和 weight 的评估
    """

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(self, batch_size: int) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    # 漂亮的打印信息
    '''
    就像如下格式：
    ╒════════╤══════════════════════════════════════╕
    │ keys   │ values                               │
    ╞════════╪══════════════════════════════════════╡
    │ l      │ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]       │
    ├────────┼──────────────────────────────────────┤
    │ l      │ [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] │
    ├────────┼──────────────────────────────────────┤
    │ l      │ [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  │
    ╘════════╧══════════════════════════════════════╛
    '''
    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    # 合并配置list
    # 就是将配置文件的内容, 加载到类中去
    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0

        # foreach key:value
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):  # 如果实例有k这个属性, 则拿到这个属性的 type, 再用设个 type 去创建一个 value, 再设置这个属性 和 属性的值
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)

if __name__ == '__main__':
    ...
    # b = BaseExp() # 抽象类,无法实例化
    # print(b)