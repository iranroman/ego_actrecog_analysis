#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import argparse
import sys
import torch
import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
from slowfast.utils.config_diff import print_compare

from test_net import test


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    og_cfg = cfg.clone()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    print_compare(og_cfg, args.cfg_file)

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main():
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform multi-clip testing.
    # ALL GPU
    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    test,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            test(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
