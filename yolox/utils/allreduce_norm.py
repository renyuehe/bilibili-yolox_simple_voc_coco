#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import pickle
from collections import OrderedDict # 顺序字典

import torch
from torch import distributed as dist # 分布式
from torch import nn

ASYNC_NORM = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

__all__ = [
    "get_async_norm_states",
    "pyobj2tensor",
    "tensor2pyobj",
    "all_reduce_norm",
]


def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


def pyobj2tensor(pyobj, device="cuda"):
    """
    serialize picklable python object to tensor
    pickle.dumps()将对象obj对象序列化并返回一个byte对象
    二进制流 转 tensor
    """
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2pyobj(tensor):
    """
    deserialize tensor to picklable python object
    pickle.loads(),从字节对象中读取被封装的对象
    """
    return pickle.loads(tensor.cpu().numpy().tobytes())

def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    所有这些都减少了不同设备中的范数统计数据。
    """
    states = get_async_norm_states(module)
    module.load_state_dict(states, strict=False)

# 自测
if __name__ == '__main__':
    # 随便一个 model
    from torchvision import models
    rn18 = models.resnet18(pretrained=True)


    # 拿到空闲的端口号
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1] # 拿到空闲的端口号
    sock.close()

    # 分布式初始化
    dist.init_process_group('gloo', init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)

    # model 设置成 并行数据
    rn18 = nn.DataParallel(rn18)

    # ★★★, 减少 归一化的。
    ret = all_reduce_norm(rn18)
    a = 10


