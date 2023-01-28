#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc

logger = logging.get_logger(__name__)


class EPICTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters, inside_action_bounds=''):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.inside_action_bounds = inside_action_bounds
        # Initialize tensors.
        self.verb_video_preds = torch.zeros((num_videos, num_cls[0]))
        self.noun_video_preds = torch.zeros((num_videos, num_cls[1]))
        if self.inside_action_bounds == 'ignore':
            # up to three sim actions per frame in validation set
            self.verb_video_labels = torch.zeros((num_videos, 3)).long()
            self.noun_video_labels = torch.zeros((num_videos, 3)).long()
        else:
            self.verb_video_labels = torch.zeros((num_videos)).long()
            self.noun_video_labels = torch.zeros((num_videos)).long()
        self.metadata = np.zeros(num_videos, dtype=object)
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.verb_video_preds.zero_()
        self.verb_video_labels.zero_()
        self.noun_video_preds.zero_()
        self.noun_video_labels.zero_()
        self.metadata.fill(0)

    def update_stats(self, preds, labels, metadata, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds[0].shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            self.verb_video_labels[vid_id] = labels[0][ind]
            self.verb_video_preds[vid_id] += preds[0][ind]
            self.noun_video_labels[vid_id] = labels[1][ind]
            self.noun_video_preds[vid_id] += preds[1][ind]
            self.metadata[vid_id] = metadata['narration_id'][ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5), inside_action_bounds=''):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        verb_topks = metrics.topk_accuracies(self.verb_video_preds, self.verb_video_labels, ks, inside_action_bounds)
        noun_topks = metrics.topk_accuracies(self.noun_video_preds, self.noun_video_labels, ks, inside_action_bounds)
        actn_topks = metrics.multitask_topk_accuracies((self.verb_video_preds,self.noun_video_preds), (self.verb_video_labels,self.noun_video_labels), ks, inside_action_bounds)

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        assert len({len(ks), len(actn_topks)}) == 1
        stats = {"split": "test_final"}
        for k, actn_topk in zip(ks, actn_topks):
            stats["action_top{}_acc".format(k)] = "{:.{prec}f}".format(actn_topk, prec=2)
        for k, verb_topk in zip(ks, verb_topks):
            stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        logging.log_json_stats(stats)
        return (self.verb_video_preds.numpy().copy(), self.noun_video_preds.numpy().copy()), \
               (self.verb_video_labels.numpy().copy(), self.noun_video_labels.numpy().copy()), \
               self.metadata.copy()
