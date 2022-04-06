#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "postprocess",
    "bboxes_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


# ★★★★★★
# 对模型的输出结果做处理, 置信度过滤, 多类别的nms过滤
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    '''
    对模型的输出结果做处理, 置信度过滤, 多类别的nms过滤,
    Args:
        prediction: 模型输出的结果
        num_classes: 类别数
        conf_thre: 置信度阈值
        nms_thre: nms 阈值
        class_agnostic: True 就做纯 nms 过滤, False 就是多类别 nms 过滤
    Returns:
    '''

    box_corner = prediction.new(prediction.shape)
    # 拿到 x1, y1, x2, y2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        # 如果没有剩余的=>进程，则处理下一个图像
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        # 以最高的置信度获得分数和类别
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()


        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask] # 置信度过滤

        if not detections.size(0):
            continue

        if class_agnostic:
            # 单类别的 nms
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            # 多类别的 nms
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6], # 类别
                nms_thre,
            )

        detections = detections[nms_out_index] # nms 过滤
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

# 同时计算 多个框a 和 多个框b 的 iou
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

# ★★★★★★
# 提供给 MosaicDetection(Dataset) 类用的
def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    # w_max, h_max 就是图片的宽高,
    # 这个 bbox 应该就是生成的框, 这些框都不能超过图片的范围,
    # 所以要通过这个函数来使框自适应图片的范围,
    # scalle_radtio 表示还原到原来的图片那么大, 因为不同的图片大小不同
    # padw 和 padh 目前还不清楚来自哪里, 起到一起偏移量的作用
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max) # * 缩放比 + 偏移量w, 区间(0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max) # * 缩放比 + 偏移量h, 区间(0, h_max)
    return bbox

# point 转 左上角point+宽高
def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

# point 转 中心点+宽高
def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
