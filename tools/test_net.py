#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import pickle
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import EPICTestMeter

logger = logging.get_logger(__name__)




def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    try:
        for cur_iter, (inputs, labels, video_idx, metadata) in enumerate(test_loader):
            # Transfer the data to the current GPU device.
            inputs = misc.to_cuda_recursive(inputs, non_blocking=True)
            labels = misc.to_cuda_recursive(labels)

            # Perform the forward pass.
            # print(misc.call_recursive(lambda x: f'{x.shape} {x.min():.4f} {x.max():.4f}', inputs))
            preds = model(inputs)
            # print(misc.call_recursive(lambda x: f'{x.shape} {x.min():.4f} {x.max():.4f}', preds))
            verb_preds, noun_preds = preds
            verb_labels, noun_labels = labels

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                _x = verb_preds, noun_preds, verb_labels, noun_labels
                verb_preds, noun_preds, verb_labels, noun_labels = du.all_gather(_x)

                meta = du.all_gather_unaligned(metadata)
                metadata = {k: [x for m in meta for x in m[k]] for k in meta[0]} if meta else {}

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                metadata, video_idx
            )
            test_meter.log_iter_stats(cur_iter)

            if cur_iter % 100 == 0:
                test_meter.compute_metrics()

            test_meter.iter_tic()
    finally:
        # Log epoch stats and print the final testing results.
        #if cfg.TEST.DATASET == 'epickitchens':
        preds, labels, metadata = test_meter.finalize_metrics(inside_action_bounds=cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS)
        #else:
        #    test_meter.finalize_metrics()
        #    preds, labels, metadata = None, None, None
        test_meter.reset()
    return preds, labels, metadata


def _load_nested_checkpoint(model, path, num_gpus):
    if not path:
        raise ValueError(f"No checkpoint for {model}")

    # recursively load checkpoints - for heterogeneous models - allow path to be a nested dict
    if isinstance(path, dict):
        for k, p in path.items():
            _load_nested_checkpoint(getattr(model, k), p, num_gpus)
        return

    # finally load the checkpoint
    cu.load_checkpoint(path, model, num_gpus > 1, None, inflation=False)


def _get_input_shape(cfg):
    video_shape = (
        cfg.DATA.IMAGE_CHANNELS,
        cfg.DATA.NUM_FRAMES,
        cfg.DATA.TRAIN_CROP_SIZE,
        cfg.DATA.TRAIN_CROP_SIZE,
    )
    audio_shape = (
        cfg.AUDIO_DATA.CHANNELS,
        cfg.AUDIO_DATA.NUM_FRAMES,
        cfg.AUDIO_DATA.NUM_FREQUENCIES,
    )
    return (
        (video_shape, audio_shape) if cfg.MODEL.VIDEO and cfg.MODEL.AUDIO else 
        video_shape if cfg.MODEL.VIDEO else 
        audio_shape
    )

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    model.eval()

    if du.is_master_proc():
        if cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, _get_input_shape(cfg), is_train=False)

    # Load a checkpoint to test if applicable.
    _load_nested_checkpoint(model, cfg.TEST.CHECKPOINT_FILE_PATH, num_gpus=cfg.NUM_GPUS)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # if slide.enable, each window plays an equal weight on the metrics
    num_clips = 1 if cfg.TEST.SLIDE.ENABLE else cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    assert len(test_loader.dataset) % num_clips == 0

    test_meter = EPICTestMeter(
        len(test_loader.dataset) // num_clips, num_clips,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS if cfg.TEST.SLIDE.ENABLE else ''
    )


    # Perform multi-view test on the entire dataset.
    preds, labels, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            results = {'verb_output': preds[0],
                       'noun_output': preds[1],
                       'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, cfg.EPICKITCHENS.TEST_SPLIT + '.pkl')
            pickle.dump(results, open(file_path, 'wb'))
