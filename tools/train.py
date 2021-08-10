#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp

import argparse
import random
import warnings


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_e2e_s-intflow_total_1K(3)")
    parser.add_argument("-n", "--name", type=str, default="yolox_e2e_intflow_s", help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=4, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='exps/intflow_total_1K/yolox_intflow_s.py',
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    #parser.add_argument("-c", "--ckpt", default='/data/pretrained/yolox_s.pth.tar', type=str, help="checkpoint file")
    #parser.add_argument("-c", "--ckpt", default='YOLOX_outputs/yolox_e2e_s-intflow_total_1K/best_ckpt.pth', type=str, help="checkpoint file")
    #parser.add_argument("-c", "--ckpt", default='YOLOX_outputs/yolox_e2e_s-intflow_total_1K(2)/best_ckpt.pth', type=str, help="checkpoint file")
    parser.add_argument("-c", "--ckpt", default='/data/pretrained/hcow/yolox_intflow_s-intflow_total_100K.pth.tar', type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=True,
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
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
