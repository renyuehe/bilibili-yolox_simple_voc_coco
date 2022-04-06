#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import functools
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
]

# 获取 cuda 总内存, 使用内存
def get_total_and_free_memory_in_Mb(cuda_device):
    # os.popen 执行 shell 命令, 拿到结果
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)

# 开辟内存挑战, 尝试能够得到多大的内存空间
def occupy_mem(cuda_device, mem_ratio=0.8):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)

# 得到 cuda 申请了多少显存
def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    跟踪一系列值，并提供对a上的平滑值的访问 窗口或全局系列的平均值
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    # 返回中位数,比如（10，20，14）返回 10
    # 如果是偶数的情况则返回 最中间的两个数的平局数,比如 (1,11,10,100) 返回 10.5
    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    # 返回平均值
    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    #
    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    # 重置
    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """
    Computes and stores the average and current value
    计算并存储平均值 和 当前值
    就是用来装 AverageMeter 类型的 字典
    """

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()


if __name__ == '__main__':
    # ret = get_total_and_free_memory_in_Mb(0)
    # print(ret)
    # ret = occupy_mem(0)
    # print(ret)
    # ret = gpu_mem_usage()
    # print(ret)

    factory = functools.partial(AverageMeter, window_size=20)
    # print(factory)
    # print(factory.median)

    abc = AverageMeter(50)
    abc.update(10)
    abc.update(11)
    abc.update(9)
    abc.update(2)
    print(abc.median)
    print(abc.avg)
    print(abc.global_avg)
    # print(abc.reset())