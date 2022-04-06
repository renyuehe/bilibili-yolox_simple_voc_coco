#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch

# 加载权重,并手动判断权重是否是ok的
def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


# 保存到检查点,如果是best,则再多考本一份并命名位best权重
def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)  #拷贝一份,内容一样,名字不一样
