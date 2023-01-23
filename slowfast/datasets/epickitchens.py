import os
import pandas as pd
import torch
import torch.utils.data
import cv2
import numpy as np

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord

from . import transform as transform
from . import utils as utils
from .frame_loader import pack_frames_to_video_clip
from .audio_loader_epic import pack_audio

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg 
        self.mode = mode
        self.target_fps = 60
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        #if self.mode in ["train", "val", "train+val"]:
        #    self._num_clips = 1
        #elif self.mode in ["test"]:
        assert self.cfg.MODEL.VIDEO or self.cfg.MODEL.AUDIO
        
        self._num_clips = (
            cfg.TEST.NUM_ENSEMBLE_VIEWS * max(1, cfg.TEST.NUM_SPATIAL_CROPS)
        )


        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        #if self.mode == "train":
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        #elif self.mode == "val":
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        #elif self.mode == "test":
        path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        #else:
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
        #                               for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._video_records.append(EpicKitchensVideoRecord(tup, idx))
        self._video_records = self._video_records[:10000]
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        record = self._video_records[index]

        if self.cfg.MODEL.VIDEO:
            video = self.get_video_clip(record)
            data = video
        if self.cfg.MODEL.AUDIO:
            audio = self.get_audio_clip(record)
            data = audio
        if self.cfg.MODEL.VIDEO and self.cfg.MODEL.AUDIO:
            data = video, audio

        return data, record.label, index, record.metadata

    def get_video_clip(self, record):
        
        #if self.mode in ["train", "val", "train+val"]:
        #    # -1 indicates random sampling.
        #    temporal_sample_index = -1
        #    spatial_sample_index = -1
        #    min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
        #    max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
        #    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        #elif self.mode in ["test"]:
        if self.mode in ["test"]:
            temporal_sample_index = (
                record.index // self.cfg.TEST.NUM_SPATIAL_CROPS)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    record.index % self.cfg.TEST.NUM_SPATIAL_CROPS)
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frames = pack_frames_to_video_clip(self.cfg, record, temporal_sample_index)

        scale = min_scale/frames.shape[1]
        if scale and scale != 1:
            frames = torch.stack([
                # The new size order for cv2.resize is w, h.
                torch.from_numpy(cv2.resize(img, (0, 0), fx=scale, fy=scale))
                for img in frames.numpy()
            ])
        
        # Perform color normalization.
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        # returns list if single/multi slowfast arch - otherwise does nothing (for omnivore and friends)
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames

    def get_audio_clip(self, record):
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = record.index
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        spec = pack_audio(self.cfg, record, temporal_sample_index, self._num_clips)
        # Normalization.
        spec = spec.float()
        if self.mode in ["train", "train+val"]:
            # Data augmentation. C T F -> C F T -> C T F
            spec = combined_transforms(spec.permute(0, 2, 1)).permute(0, 2, 1)

        spec = utils.pack_pathway_output(self.cfg, spec)
        return spec

    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            #assert len({min_scale, max_scale, crop_size}) == 1
            #frames, _ = transform.random_short_side_scale_jitter(
            #    frames, min_scale, max_scale
            #)
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
