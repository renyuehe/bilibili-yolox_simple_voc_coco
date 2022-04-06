#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        # 动态导入模块 exp_file ==>> yolox_xxx
        sys.path.append(os.path.dirname(exp_file)) # exp_file 添加系统环境变量,添加了路径才能导入
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0]) # 等价于 import “yolox_xxx"等价于 import “yolox_xxx"

        exp = current_exp.Exp() # 实例化模块中的 Exp类

    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp

def get_exp_by_name(exp_name):
    import yolox
    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__)) # 上跳两级==>>根目录

    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolov3": "yolov3.py",
    }

    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)

    return get_exp_by_file(exp_path)

def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name.
    通过文件或名称获取exp对象。
    Args:
        exp_file (str): file path of experiment. 实验文件路径
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
