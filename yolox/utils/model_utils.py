#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
from copy import deepcopy


import torch
import torch.nn as nn
from thop import profile


__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
]


# 获取模型的信息(参数量 + 运算量)
def get_model_info(model, tsize):
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6  # 参数量
    flops /= 1e9   # 浮点运算量
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    '''
    Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    融合 卷积层 和 batchnormal层
    Args:
        conv:更新卷积
        bn:删除bn
    Returns:

    '''

    # 对原卷积核,偏移量变true,不寻求梯度
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True, # 偏移量设置位 true
        )
        .requires_grad_(False) # 不允许求导
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    # 这里就是对 weight 级别 conv 和 bn 层的融合
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )

    # 这里就是对 bias 级别 conv 和 bn 层的融合
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    '''
    # 融合模型(压缩模型)
    Args:
        model:
    Returns:返回一个模型
    '''
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        # 对 BaseConv 模块进行Conv层和BN层的融合
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward

    return model


# 这个函数有点没有懂, 表面上看起来是model的替换, 但是参数为什么传入的 Silu
def replace_module(module,  replaced_module_type,  new_module_type,  replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.
    将模块中的给定类型替换为新类型。                  主要用于部署。

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


if __name__ == '__main__':
    os.chdir("../../")
    from loguru import logger
    from yolox.exp import get_exp

    ####### 获取模型信息
    exp_file = r'exps/example/yolox_voc/yolox_voc_s.py'
    name = "yolox-voc-s"
    exp = get_exp(exp_file, name)
    model = exp.get_model()
    logger.info(
        "Model Summary: {}".format(get_model_info(model, exp.test_size))
    )

    ######