# 补充
import os, sys
sys.path.append(os.path.abspath(os.path.curdir))

# 正文
import argparse
import random
import warnings
from loguru import logger


import torch.backends.cudnn as cudnn

from yolox.core import Trainer
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")


    # parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    # parser.add_argument("-expn", "--experiment-name", type=str, default=r"yolox_s_coco128/pth_dir")
    parser.add_argument("-expn", "--experiment-name", type=str, default=r"yolox_voc_s/pth_dir")

    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        # default=r"exps/default/yolox_s.py",
        # default=r"exps/example/custom/yolox_s.py",
        default = r"exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="plz input your experiment description file",
    )

    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-c", "--ckpt", default="./yolox_s.pth", type=str, help="checkpoint file")
    # parser.add_argument("-c", "--ckpt", default=r"YOLOX_outputs/yolox_s_coco128/pth_dir/last_epoch_ckpt.pth", type=str, help="checkpoint file")
    # parser.add_argument("-c", "--ckpt", default=r"YOLOX_outputs/yolox_voc_s/pth_dir/best_ckpt.pth", type=str, help="checkpoint file")

    # 恢复训练
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )

    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")

    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )

    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )

    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    '''
    设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    应该遵循以下准则：
    如果网络的输入数据维度或类型上变化不大，也就是每次训练的图像尺寸都是一样的时候，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率；
    如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    '''
    cudnn.benchmark = False

    # ★★★ 改代码的核心
    # 创建一个训练器
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 获取 arg
    args = make_parser().parse_args()

    # 获取 exp
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts) # 合并exp配置

    # 如果没有填写获取 experiment 默认参数
    args.experiment_name = exp.exp_name if not args.experiment_name else args.experiment_name

    main(exp, args)