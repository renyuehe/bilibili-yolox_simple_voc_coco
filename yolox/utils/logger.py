#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import inspect
import os
import sys
from loguru import logger


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.
        深度(int)：调用者对话的深度，使用0用于调用者的深度。默认值：0。
    Returns:
        str: module name of the caller
        str: 调用者的模块名称
    """
    # the following logic is a little bit faster than inspect.stack() logic
    # 下面的逻辑比检查。stack()逻辑要快一点
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"] # 返回调用的模块名


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    流对象 将写入 重定向到 记录器实例 。
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            level : loguru的日志级别字符串。默认值 "INFO"
            caller_names(tuple): caller names of redirected module.
            caller_names : 重定向模块的调用者名称
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1) # 拿到调用者模块
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


if __name__ == '__main__':
    print(sys.stderr)
    print(sys.stdout)
    redirect_sys_output()
    print(sys.stderr)
    print(sys.stdout)


def setup_logger(save_dir, filename="log.txt", mode="a"):
    """ setup logger for training and testing.
        为培训和测试设置记录器。
    Args:
        save_dir(str): location to save log file
        save_dir(str): 要保存日志文件的位置。

        distributed_rank(int): device rank when multi-gpu environment
        distributed_rank(int): 多gpu环境下的设备排名。

        filename (string): log save name.
        filename (string): 日志保存名称。

        mode(str): log file write mode, `append` or `override`. default is `a`.
        mode(str): 日志文件写入模式，`附加` 或 `覆盖`。默认为 `a`。

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    logger.add(
        sys.stderr,
        format=loguru_format,
        level="INFO",
        enqueue=True,
    )
    logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")
