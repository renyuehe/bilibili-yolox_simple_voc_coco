#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.

    YOLOX模型模块。
    模块列表由 create_yolov3_modules 函数定义。
    网络在训练期间返回三个YOLO层的损失值 以及测试过程中的检测结果。
    """

    # 主干网络 和 侦测头 可以自定义
    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    # 推理（x 是 网络的输入, targets 是 标签）
    # 训练的时候返回损失, 测试的时候返回前向结果
    # x, 输入:        [batch_size, channel, h, w]
    # targets,标签:   [batch_size, 120, 85]
    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        # 主干网络输出, 默认主干网络是 YOLOPAFPN
        fpn_outs = self.backbone(x)

        # 训练状态,返回损失
        if self.training:
            assert targets is not None

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)

            outputs = {
                "total_loss": loss, # 总损失
                "iou_loss": iou_loss, # iou 损失
                "l1_loss": l1_loss, # ？？？
                "conf_loss": conf_loss, # 置信度损失
                "cls_loss": cls_loss, # 类别
                "num_fg": num_fg, # ？？？
            }
        # 测试状态,返回输出
        else:
            outputs = self.head(fpn_outs)

        return outputs

if __name__ == '__main__':
    yolox = YOLOX()
    print(yolox)