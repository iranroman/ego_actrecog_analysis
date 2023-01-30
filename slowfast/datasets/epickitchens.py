import os
import pandas as pd
import torch
import torch.utils.data
import cv2
import numpy as np
import datetime

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord, timestamp_to_sec

from . import transform as transform
from . import utils as utils
from .frame_loader import pack_frames_to_video_clip
# from .audio_loader_epic import pack_audio
from .audio_loader_epic_og import pack_audio

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
        if self.mode == "train":
           path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
           path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
           path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                                      for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(file)

        self._video_records = []
        # for file in path_annotations_pickle:
        #     for tup in pd.read_pickle(file).iterrows():
        #         for idx in range(self._num_clips):
        #             self._video_records.append(EpicKitchensVideoRecord(tup, idx))
        # # self._video_records = self._video_records[:10000]
        iii = 0
        video_durs = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VIDEO_DURS))
        video_durs = dict(zip(video_durs['video_id'],video_durs['duration']))
        if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'ignore':
            video_times = {}
        for file in path_annotations_pickle:
            for narr_id, row in pd.read_pickle(file).iterrows():
                if not self.cfg.TEST.SLIDE.ENABLE:
                    for idx in range(self._num_clips):
                        self._video_records.append(EpicKitchensVideoRecord((narr_id, row), idx))
                else:
                    action_start_sec = timestamp_to_sec(row['start_timestamp'])
                    action_stop_sec = timestamp_to_sec(row['stop_timestamp'])
                    _win_sec = self.cfg.TEST.SLIDE.HOP_SIZE*np.ceil(action_start_sec/self.cfg.TEST.SLIDE.HOP_SIZE)
                    if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'strict':
                        win_start_sec, win_stop_sec = _win_sec, _win_sec + self.cfg.TEST.SLIDE.WIN_SIZE
                    elif self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'outside':
                        win_start_sec, win_stop_sec = _win_sec - self.cfg.TEST.SLIDE.WIN_SIZE, _win_sec
                    elif self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'ignore':
                        win_label_sec = _win_sec
                        win_start_sec = win_label_sec - self.cfg.TEST.SLIDE.LABEL_FRAME*self.cfg.TEST.SLIDE.WIN_SIZE
                        win_stop_sec = win_start_sec + self.cfg.TEST.SLIDE.WIN_SIZE
                    win_start_sec = max(0, win_start_sec)

                    while (
                        (win_stop_sec < action_stop_sec) if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'strict' else 
                        (win_start_sec < action_stop_sec) if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'outside' else 
                        (win_label_sec < action_stop_sec)
                    ):
                        ek_ann = row.copy()
                        ek_ann['start_timestamp'] = (datetime.datetime.min + datetime.timedelta(seconds=win_start_sec)).strftime('%H:%M:%S.%f')
                        ek_ann['stop_timestamp'] = (datetime.datetime.min + datetime.timedelta(seconds=win_stop_sec)).strftime('%H:%M:%S.%f')
                        win_stop_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                        win_start_sec = max(0, win_stop_sec - self.cfg.TEST.SLIDE.WIN_SIZE)
                        if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'ignore':
                            # up to three sim actions per frame in validation set
                            ek_ann['verb_class'] = [ek_ann['verb_class']] * 3
                            ek_ann['noun_class'] = [ek_ann['noun_class']] * 3
                            ek_ann['verb'] = [ek_ann['verb']] * 3
                            ek_ann['noun'] = [ek_ann['noun']] * 3
                            video_timestamp = f'{ek_ann["video_id"]}_{win_label_sec:.2f}'
                            win_label_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                            if video_timestamp in video_times:
                                video_record = self._video_records[video_times[video_timestamp]]
                                unique_verb_noun = list(set(zip(video_record._series['verb_class'],video_record._series['noun_class'])))
                                if (ek_ann["verb"],ek_ann["noun"]) not in unique_verb_noun:
                                    video_record._series['verb_class'][len(unique_verb_noun)] = ek_ann['verb_class'][0]
                                    video_record._series['noun_class'][len(unique_verb_noun)] = ek_ann['noun_class'][0]
                                    video_record._series['verb'][len(unique_verb_noun)] = ek_ann['verb'][0]
                                    video_record._series['noun'][len(unique_verb_noun)] = ek_ann['noun'][0]
                                self._video_records[video_times[video_timestamp]] = video_record
                                continue
                            
                            video_times[video_timestamp] = iii

                        self._video_records.append(EpicKitchensVideoRecord((narr_id, ek_ann), 0))
                        self._video_records[-1].time_end = video_durs[ek_ann['video_id']]
                        iii += 1

        if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS == 'ignore':
            for iii in range(len(self._video_records)):
                self._video_records[iii]._series['noun_class'] = np.array(self._video_records[iii]._series['noun_class'])
                self._video_records[iii]._series['verb_class'] = np.array(self._video_records[iii]._series['verb_class'])

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
        self.audio_dataset = None

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
        if self.audio_dataset is None and self.cfg.EPICKITCHENS.AUDIO_DATA_FILE:
            import h5py
            self.audio_dataset = h5py.File(self.cfg.EPICKITCHENS.AUDIO_DATA_FILE, 'r')

        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = record.index
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # spec = pack_audio(self.cfg, record, temporal_sample_index, self._num_clips, h5_reader=self.audio_dataset)
        spec = pack_audio(self.cfg, self.audio_dataset, record, temporal_sample_index)
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
