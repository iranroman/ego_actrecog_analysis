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

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        video_idx = video_idx.cuda()

        # Perform the forward pass.
        #print(inputs[0,:,0,0])
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            verb_preds, verb_labels, video_idx = du.all_gather(
                [preds[0], labels['verb'], video_idx]
            )

            noun_preds, noun_labels, video_idx = du.all_gather(
                [preds[1], labels['noun'], video_idx]
            )
            meta = du.all_gather_unaligned(meta)
            metadata = {'narration_id': []}
            for i in range(len(meta)):
                metadata['narration_id'].extend(meta[i]['narration_id'])
        else:
            metadata = meta
            verb_preds, verb_labels, video_idx = preds[0], labels['verb'], video_idx
            noun_preds, noun_labels, video_idx = preds[1], labels['noun'], video_idx
        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
            (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
            metadata,
            video_idx.detach().cpu(),
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    #if cfg.TEST.DATASET == 'epickitchens':
    preds, labels, metadata = test_meter.finalize_metrics(per_action_instance=cfg.TEST.SLIDE.PER_ACTION_INSTANCE)
    #else:
    #    test_meter.finalize_metrics()
    #    preds, labels, metadata = None, None, None
    test_meter.reset()
    return preds, labels, metadata

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
    if du.is_master_proc():
        if cfg.MODEL.MODEL_NAME == 'SlowFast':
            misc.log_model_info(model, cfg, is_train=False)
        # TODO: make it work with Omnivore and TSM

    # Load a checkpoint to test if applicable.
    if cfg.MODEL.MODEL_NAME == 'SlowFast':
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cfg.MODEL.MODEL_NAME == 'TSM':
        [cu.load_checkpoint(
            f'{cfg.TEST.CHECKPOINT_FILE_PATH}_{t}.ckpt',
            getattr(model,f'base_model_{t}'),
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        ) for t in ['rgb','flow']]

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if not cfg.TEST.SLIDE.ENABLE:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )


    # Create meters for multi-view testing.
    #if cfg.TEST.DATASET == 'epickitchens':
    if not cfg.TEST.SLIDE.ENABLE:
        test_meter = EPICTestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            weight_instance_by_nframes=cfg.TEST.WEIGHT_INSTANCE_BY_NFRAMES,
        )
    else:
        test_meter = EPICTestMeter(
            len(test_loader.dataset),
            1, # each window plays an equal weight on the metrics
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.TEST.SLIDE.PER_ACTION_INSTANCE,
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
