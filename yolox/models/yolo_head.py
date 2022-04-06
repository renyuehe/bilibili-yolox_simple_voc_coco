#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math

import numpy as np
from loguru import logger

import torch
import torch.nn as nn
from torch.nn import functional

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes, # 类别数
        width=1.0, # 网络宽度
        strides=[8, 16, 32], # 尺寸缩放比例, 640/8=80, 640/16=40, 640/32=20
        in_channels=[256, 512, 1024], # 输入通道, 对应不同尺寸，应与 strides 数量相同
        act="silu", # 激活函数
        depthwise=False, # 是否 depthwise(深度可分离卷积) + conv
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            act (str)：conv的激活类型。Defalut值：“silu”。
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
            depthwise (bool)：是否将 conv 替换为 DWConv。Defalut值：False。
        """
        super().__init__()

        # !----------------------- ??? -----------------------!#
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False 若要进行部署，请设置为False

        # !----------------------- 配置网络结构 -----------------------!#
        self.stems = nn.ModuleList() # 分别对应不同尺寸的 主干

        self.cls_convs = nn.ModuleList() # 分别对应不同尺寸的 一级解耦 cls
        self.cls_preds = nn.ModuleList() # 分别对应不同尺寸的 二级解耦 cls

        self.reg_convs = nn.ModuleList() # 分别对应不同尺寸的 一级解耦 reg_conv
        self.reg_preds = nn.ModuleList() # 分别对应不同尺寸的 二级解耦 reg
        self.obj_preds = nn.ModuleList() # 分别对应不同尺寸的 二级解耦 obj

        Conv = DWConv if depthwise else BaseConv

        # 有几个输入通道, 就有几个 (cls、ret、obj)
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        #!----------------------- 配置损失 -----------------------!#
        # ★★★ 难点, 训练时候为了提高效果, 特制的一个 L1 关于 ??? 的损失
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

        # !----------------------- ??? -----------------------!#
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels) # 存放不同尺度的坐标索引

    # 整个项目还没有用到
    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # 遍历每一个尺度, 分别forward每一个尺度, 并做简单处理
        # 尺度体现在 in_channel 上, 尺度的数量应该和 strides数量相同
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x) # head 主干
            cls_x = x # cls 开始解耦
            reg_x = x # reg 开始解耦

            # 类别解耦头
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat) # 输出类别

            # 回归框 和 置信度解耦头
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat) # 回归框再次解耦, 输出回归狂
            obj_output = self.obj_preds[k](reg_feat) # 置信度再次解耦, 输出置信度

            # 训练模式
            if self.training:
                # ★ 拿到网络所有解耦头的输出并合并
                output = torch.cat([reg_output, obj_output, cls_output], dim=1) # 通道层将 reg,obj,cls 粘起来

                # ★ 获取 输出    output (batch_size, xxx, 通道特征), 比如 (1, 6400, 85)
                # ★ 获取 网格索引 grid   (batch_zie, xxx, xy索引), 比如 (1, 6400, 2)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )


                x_shifts.append(grid[:, :, 0]) # x 相对与全图的偏移量
                y_shifts.append(grid[:, :, 1]) # y 相对与全图的偏移量

                # ??? 目的不明
                # 网格_填充_尺寸缩放比, 比如: (1, 6400, 8)
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]) # 就是网格大小(6400)
                    .fill_(stride_this_level) # 填充值: 缩放比例
                    .type_as(xin[0])
                )

                # 就是专门对回归框做的一个 损失分支 前 回归框数据准备
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]

                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )

                    origin_preds.append(reg_output.clone())
            # 测试模式
            else:
                # ★ 拿到网络所有解耦头的输出并合并
                # .sigmoid() 是因为前面使用的 BCEWithLogitsLoss
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1 # 默认
                    # [reg_output, obj_output, cls_output], 1 # 测试改动
                )

            # 多个尺度的输出汇总
            outputs.append(output)

        if self.training:
            # 直接输出所有分支的损失
            return self.get_losses(
                imgs = imgs, # 输入的原始 batch_size数据
                x_shifts = x_shifts, # x轴关于全图的偏移量
                y_shifts = y_shifts, # y轴关于全图的偏移量
                expanded_strides = expanded_strides, # ???
                labels = labels, # 标签
                outputs = torch.cat(outputs, 1), # 输出
                origin_preds = origin_preds, # 回归框 l1损失 特殊分支
                dtype  =xin[0].dtype, # 数据类型
            )
        else:
            # 拿到 (h,w) 信息
            self.hw = [x.shape[-2:] for x in outputs]

            # ★★★
            # [list:3 [batch, 85, √n_anchors, √n_anchors]]
            # ==>>
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)


            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    # 获取输出(输出的前4个通道 reg框通道, 变成了相对于全图的通道), 和网格索引
    def get_output_and_grid(self, output, k, stride, dtype):
        '''
        # ★ 获取 输出    output (batch_size, xxx, 通道特征), 比如 (1, 6400, 85)
        # ★ 获取 网格索引 grid   (batch_zie, xxx, xy索引), 比如 (1, 6400, 2)
        Args:
            output:[batch_size, 85, hsize, wsize]
            k:当前尺度.(0,1,2,...)
            stride:当前尺度对应的缩放比(8, 16, 32)
            dtype:

        Returns:
        '''
        grid = self.grids[k]

        batch_size = output.shape[0] # batch_size
        n_ch = 5 + self.num_classes # 获取通道数 85
        hsize, wsize = output.shape[-2:] # hsize, wsize

        # 如果 grid 为空, 则生成 grid, (坐标索引)
        if grid.shape[2:4] != output.shape[2:4]:
            # 生成y轴网格, 生成x轴网格。 meshgrid(网格)
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # 造坐标索引
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype) # [1, 1, hsize, wsize, 2], 这个结果就是
            self.grids[k] = grid # 将 当前k(hsize, wsize)尺度的坐标索引存放在当前尺度grids[k]中

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)

        # 先 ==>> (batch_size, self.n_anchors, hsize, wsize, n_ch)
        # 再 ==>> (batch_size, xxx, n_ch)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        # (1, xxx, 2) == (batchsize, xxx, xy索引)
        grid = grid.view(1, -1, 2)

        # 对置信度进行操作
        # reg框的前2个参数表示: 相对于全图的偏移量,（grid整数索引+小数索引） * 尺寸还原
        # reg框的后2个参数表示: w 和 h 的偏移量, (exp(h偏移量, w偏移量)) * 尺寸还原
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    # ???
    # 只有在非部署的时候调用, 部署的时候是不能调用的
    # 对 outputs 的一个包装
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs


    def get_losses(
        self,
        imgs, # 原始batch_size输入数据
        x_shifts, # x全图偏移
        y_shifts, # y全图偏移
        expanded_strides, # 网格_填充_尺寸缩放比
        labels, # 标签
        outputs, # outputs [batch, n_anchors_all, 4+1+n_cls], 比如[batch, 6400+1600+400, 85]
        origin_preds, # l1损失 回归框扩充分支
        dtype,
    ):
        # bbox
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets 算出有多少个标签label
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # 多尺寸 * n_anchors * 网格总数 个 anchor
        # 总共的 anchor 数
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]

        expanded_strides = torch.cat(expanded_strides, 1) # 对应每个 anchor 的尺寸系数

        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0 # 所有image批次的 预测框 和
        num_gts = 0.0 # 所有image批次的 ground truth 和

        # 遍历批次
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt

            if num_gt == 0: # 如果没有 ground truth
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else: # 如果有 ground truth
                # 标签类, 标签box
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5] # 拿到所有标签中的 框
                gt_classes = labels[batch_idx, :num_gt, 0] # 拿到所有标签中的 类
                # 预测 box
                bboxes_preds_per_image = bbox_preds[batch_idx] # 拿到预测的 boxxes

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()


                    # ★★★ 拿到 dynamic K, 动态 k 框
                    (
                        gt_matched_classes,         # dynamic k个框 对应的类别
                        fg_mask,                    # 所有尺寸中 符合条件的 网格预测框的 mask
                        pred_ious_this_matching,    # dynamic k个框 对应的ious
                        matched_gt_inds,            # dynamic k个框 对应的 gt框 索引
                        num_fg_img,                 # dyanmic k
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )
                torch.cuda.empty_cache()

                num_fg += num_fg_img

                cls_target = functional.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)


        num_fg = max(num_fg, 1)

        # 回归框采用 iou loss
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg

        # 置信度 采用 BCEWithLogitsLoss
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        # 类别 loss 采用 BCEWithLogitsLoss
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1), # 所有批次的 dynamic_k框 和 / 所有批次的 ground_truth框 和
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    # simOTA流程
    # 最终获得动态 dynamic k, 相对于gt框的生成的动态k框 ★★★★★★
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):
        '''
        return
        (
            gt_matched_classes,         # dynamic k个框 对应的类别
            fg_mask,                    # 所有尺寸中 符合条件的 网格预测框的 mask
            pred_ious_this_matching,    # dynamic k个框 对应的ious
            matched_gt_inds,            # dynamic k个框 对应的 gt框 索引
            num_fg,                     # dyanmic k
        )
        '''
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # 筛选出符合框条件的,中心点符合条件 and 框值符合条件
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,#所有标签(num_gt, cls, bbox)==(num_gt, 1, 4)
            expanded_strides,#尺寸系数
            x_shifts, #网格x索引
            y_shifts, #网格y索引
            total_num_anchors,#
            num_gt,#标签数量
        )

        # 筛选出来符合条件的 每张图 bboxes
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # 筛选出来符合条件的 每张图 cls
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        # 筛选出来符合条件的 每张图 obj
        obj_preds_ = obj_preds[batch_idx][fg_mask]

        # 符合条件的 anchor 数
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # 同时计算多个框和多个框的 iou
        # 通过 gt 框 和 三个尺寸中所有符合条件的框 做 iou
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        # shape ==>> [1, num_in_boxes_anchor, num_classes]
        gt_cls_per_image = (
            functional.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        # 拿到 置信度 的损失
        # iou损失, 用 iou 和 信息量 来做
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()


        with torch.cuda.amp.autocast(enabled=False):
            # sigmoid(置信度概率s * 类别概率s), 更抽象的表示类别的置信度?
            # sigmoid((num_gt, num_in_boxes_anchor, num_cls))
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            # 拿到分类的损失
            # 分类问题损失 用交叉熵损失
            pair_wise_cls_loss = functional.binary_cross_entropy(
                # 开根号是因为 (cls的概率 * 置信度概率) 相乘了
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

        del cls_preds_

        cost = (
            pair_wise_cls_loss # 分类损失
            + 3.0 * pair_wise_ious_loss # 置信度损失
            + 100000.0 * (~is_in_boxes_and_center) # 为了让损失减小, 负样本都要趋向与 0 才可以
        )

        (
            num_fg, # 动态k个,网格预测ok框数
            gt_matched_classes,# 这k个框对应的类别
            pred_ious_this_matching, # 这k个框对应的ious
            matched_gt_inds, # 这k个框对用的 gt框 索引
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes, # dynamic k个框 对应的类别
            fg_mask, # 所有尺寸中 符合条件的 网格预测框的 mask
            pred_ious_this_matching, # dynamic k个框 对应的ious
            matched_gt_inds, # dynamic k个框 对应的 gt框 索引
            num_fg, # dyanmic k
        )


    # 筛选符不符合框条件的
    # return is_in_boxes_or_anchor, is_in_boxes_and_center
    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        # 每张图的多尺度缩放系数
        expanded_strides_per_image = expanded_strides[0]
        # 每张图的网格x轴索引 * 缩放系数 == 原图 x轴 索引, 但不是中心点
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        # 每张图的网格y轴索引 * 缩放系数 == 原图 y轴 索引, 但不是中心点
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        # 将 原图x轴索引,偏移至中心点, 并repeat成num_gt份 ???
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        # 将 原图y轴索引,偏移至中心点, 并repeat成num_gt份 ???
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        # 复查处, 这里的算法说明 wh 但是真的是这样吗？我前面的记录写的 hw是错的吗？ 需要验证
        # 中心点 - 0.5宽 = left
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 中心点 + 0.5宽 = right
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 中心点 - 0.5高 = top
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 中心点 + 0.5高 = button
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 左右上下相对于中心点的偏移量
        # 特点大的减小的,保证了都为正
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # 合并,(左上右下)的偏移量,shape(num_gt,pred_anchors_all,4)
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # 如果(左上右下)的偏移量 全都 > 0.0 就表示有一个框确实是一个框。 (num_gt, hw_anchors_all) bool矩阵
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 #
        # num_gt 个标签框（怎么看起来像建议框的作用?) 只要有一个框是 True ,则网格xy索引位置成立。
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        #!------------- 上面的是中心点半径根据 宽高 计算 left,right,top,button -------------！#
        #!------------- 下面的是中心点半径固定 2.5 计算 left,right,top,button -------------！#
        # in fixed center
        # 中心点半径固定为 2.5
        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image

        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        # boxes或centers
        is_in_boxes_or_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_or_anchor] & is_in_centers[:, is_in_boxes_or_anchor]
        )

        return is_in_boxes_or_anchor, is_in_boxes_and_center

    # ★★★★★★
    def dynamic_k_matching(
        self,
        cost, # 前面计算的 loss
        pair_wise_ious, # gt框 和 符合条件的框 的iou矩阵
        gt_classes, #
        num_gt, #
        fg_mask # 网格中 符合条件的框 的 mask
    ):
        # Dynamic K
        matching_matrix = torch.zeros_like(cost) # 生成和cost形状类型一直的张量

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1)) # 动态k这里设置默认为10,★★★
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1) # 在第 1 个维度上截取 n_candidate_k 个值最高的值,并返回
        # 动态力度, iou越大, 表示效能越大, sum加和后就越强, 则动态力度就越大
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)# k框概率加和取整,最小也为1
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], # gt框 和 网格 iou损失矩阵, 取当前gt的那一个个向量
                k=dynamic_ks[gt_idx].item(), # ★★★★★★ 动态k框的动态力度  这个力度才是真正绝妙的地方, 这是第二层动态
                dim=-1, # 最后一个维度
                largest=False # 表示 topk 取值最小的 top, 换句话说就是损失越小,代表这个框越 ok
            )
            matching_matrix[gt_idx][pos_idx] = 1.0 # ★★★★★★ 于是 gt_idx 找到了 动态k框的动态力度, 将位置索引置为 1

        del topk_ious, dynamic_ks, pos_idx

        # ??? 为什么要sum0
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:  #这里就是把 matching_matrix 矩阵中>1的值,设置为1
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # 拿到网格预测框ok的mask
        num_fg = fg_mask_inboxes.sum().item() # 有多少个网格预测ok框

        fg_mask[fg_mask.clone()] = fg_mask_inboxes # fg_mask 中为 True 的部分 再按照顺序 设置为 fg_mask_inboxes 中的 True、False 顺序

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) # 找到 gt框 最相近的 网格预测ok框 的索引
        gt_matched_classes = gt_classes[matched_gt_inds]
        # 网格预测ok框的ious矩阵
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
