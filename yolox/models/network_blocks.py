#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    '''
    字符串反射方激活函数
    只支持 silu, relu, lrelu
    '''
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# Conv2d + Batchnorm + 激活函数 的组合模块,
# 其中特性是 padding 自适应, 保证了 kernel_size 任意时 padding 不改变
class BaseConv(nn.Module):
    """
    A Conv2d -> Batchnorm -> silu/leaky relu block
    """

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 # pad 的计算是为了保证 conv2d 之后, 宽高可以保持不变

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        # 反射出 激活函数, 支持 silu, relu, lrelu
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) # 卷积 + batchnormal + 激活

    # 该函数是 融合模型(fuse_model)的时候, 用来替代 forward的
    def fuseforward(self, x):
        return self.act(self.conv(x)) # 融合后的前向, 就是少了 batchnormal

# MobileNetV1 (操作像素的时候不操作通道, 操作通道的时候不操作像素, 速度快)
class DWConv(nn.Module):
    """
    Depthwise Conv + Conv
    深度可分离卷积 + 卷积 == MobileNetV1 (操作像素的时候不操作通道, 操作通道的时候不操作像素, 速度快)

    深度可分离卷积: 只做像素融合,不做通道融合（速度快）
    Conv 1*1: 只做通道融合, 不做像素融合(速度快)
    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels, # ★★★ 组数等于输入通道, 通道不融合
            act=act,
        )

        self.pconv = BaseConv(
            in_channels,
            out_channels,
            ksize=1, # ★★★ 1 * 1, 像素不融合
            stride=1,
            groups=1,
            act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

# 瓶颈结构
class Bottleneck(nn.Module):
    '''
    Standard bottleneck
    瓶颈模型: 前半截通道融合, 后半截通道融合+像素融合, 既保证了特征提取,又保证了像素上和通道上的融合
    '''

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True, # 当 in_channel == out_chacnnel 的时候, 也就是瓶颈结构对称的时候, 是否使用残差模块
        expansion=0.5, # 中间层压缩系数
        depthwise=False, # 是否使用 depthwise conv 代替 conv
        act="silu", # 激活函数
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion) # 瓶颈结构中间层通道
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=act) # 瓶颈结构前半截
        self.conv2 = Conv(hidden_channels, out_channels, ksize=3, stride=1, act=act) # 瓶颈结构后半截

        self.use_add = shortcut and in_channels == out_channels # 是否使用残差

    def forward(self, x):
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y

# 残差层, 类似于对称瓶颈模型, 中间层固定压缩一半
# 前半截通道融合,后半截通道融合+像素融合
class ResLayer(nn.Module):
    '''
    Residual layer with `in_channels` inputs.
    残差层, 类似于对称瓶颈模型, 中间层固定压缩一半
    前半截通道融合, 后半截通道融合+像素融合
    '''

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2

        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


# 中间采用空间金字塔, 金字塔 cat 后再约束到 output_channel
class SPPBottleneck(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP
    YOLOv3-SPP中使用的空间金字塔池化层
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(5, 9, 13), # 池化 kernel_size 以及池化几次
        activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        # 不做像素融合
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=activation)

        # 该空间金字塔池化层不改变 w 和 h
        self.m = nn.ModuleList(
            # stride==None的时候,默认为kernel_size, 如果指定 stride为1, 则就和卷积一样了
            # padding = ks // 2 也是保证了 w,h不会改变
            [ nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes ]
        )

        # 空间金子塔 cat 后的 channel
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)

        # 不做像素融合
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1) # 把空间金字塔的每一层都在通道级 cat 起来
        x = self.conv2(x)
        return x

# csp多瓶颈结构 + 1*1卷积
# csp多瓶颈结构:较强的语义特征提取能力
# 1*1卷积:通道层面特征提取
class CSPLayer(nn.Module):
    """
    C3 in yolov5, CSP Bottleneck with 3 convolutions
    C3 in yolov5, CSP瓶颈与3个卷积
    语义特征提取能力强 + 通道融合卷积
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1, # 多少个 Bottleneck
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # CSP瓶颈, 像素融合 和 通道融合都有, 且有较强语义提取的功能
        self.m = nn.Sequential(
            # 多个带残差的 Bottleneck
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        )

        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.m(x_1)# CSP瓶颈, 像素融合 和 通道融合都有, 且有较强语义提取的功能

        x_2 = self.conv2(x) # 1*1 像素融合,通道融合卷积

        x = torch.cat((x_1, x_2), dim=1) # 粘在一起
        return self.conv3(x)


# 聚焦到四个区域：top_left、top_right、bot_left、bot_right
# 输入 (b,c,w,h) -> 输出 (b,4c,w/2,h/2)
class Focus(nn.Module):
    """
    Focus width and height information into channel space.
    将宽度和高度信息聚焦到通道空间中。
    聚焦到四个区域：top_left、top_right、bot_left、bot_right
    输入 (b,c,w,h) -> 输出 (b,4c,w/2,h/2)
    再做卷积
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=1,
        stride=1,
        act="silu"
    ):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )

        return self.conv(x)
