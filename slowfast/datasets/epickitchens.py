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
from .frame_loader import pack_frames_to_video_clip, pack_flow_frames_to_video_clip

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
        if not cfg.TEST.SLIDE.ENABLE:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []
        iii = 0

        # get the video duration
        video_durs = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VIDEO_DURS))
        video_durs = dict(zip(video_durs['video_id'],video_durs['duration']))
        if not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
            video_times = {}
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                if not self.cfg.TEST.SLIDE.ENABLE:
                    for idx in range(self._num_clips):
                        self._video_records.append(EpicKitchensVideoRecord(tup))
                        self._spatial_temporal_idx.append(idx)
                        self._video_records[-1].time_end = video_durs[tup[1]['video_id']]
                else:

                    # get the action start and end
                    action_start_sec = timestamp_to_sec(tup[1]['start_timestamp'])
                    action_stop_sec = timestamp_to_sec(tup[1]['stop_timestamp'])

                    if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
                        win_start_sec = self.cfg.TEST.SLIDE.HOP_SIZE*np.ceil(action_start_sec/self.cfg.TEST.SLIDE.HOP_SIZE)
                        win_stop_sec = win_start_sec + self.cfg.TEST.SLIDE.WIN_SIZE
                        win_label_sec = (win_stop_sec-win_start_sec)/2
                    else:
                        win_label_sec = self.cfg.TEST.SLIDE.HOP_SIZE*np.ceil(action_start_sec/self.cfg.TEST.SLIDE.HOP_SIZE)
                        win_start_sec = win_label_sec - self.cfg.TEST.SLIDE.LABEL_FRAME*self.cfg.TEST.SLIDE.WIN_SIZE
                        win_stop_sec = win_label_sec + (1-self.cfg.TEST.SLIDE.LABEL_FRAME)*self.cfg.TEST.SLIDE.WIN_SIZE
                    win_start_sec = win_start_sec if win_start_sec > 0.0  else 0.0
                    while (win_label_sec < action_stop_sec):
                        ek_ann = tup[1].copy()
                        ek_ann['start_timestamp'] = (datetime.datetime.min + datetime.timedelta(seconds=win_start_sec)).strftime('%H:%M:%S.%f')
                        ek_ann['stop_timestamp'] = (datetime.datetime.min + datetime.timedelta(seconds=win_stop_sec)).strftime('%H:%M:%S.%f')
                        ek_ann['action_stop_frame'] = action_stop_sec * self.target_fps
                        if not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
                            # up to three sim actions per frame in validation set
                            ek_ann['verb_class'] = [ek_ann['verb_class']] * 3
                            ek_ann['noun_class'] = [ek_ann['noun_class']] * 3
                            ek_ann['verb'] = [ek_ann['verb']] * 3
                            ek_ann['noun'] = [ek_ann['noun']] * 3
                            video_timestamp = f'{ek_ann["video_id"]}_{win_label_sec:.2f}'
                            win_label_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                            if video_timestamp not in video_times:
                                video_times[video_timestamp] = iii
                                self._video_records.append(EpicKitchensVideoRecord((tup[0],ek_ann)))
                                self._video_records[-1].time_end = video_durs[ek_ann['video_id']]
                                self._spatial_temporal_idx.append(0)
                                iii += 1
                            else:
                                video_record = self._video_records[video_times[video_timestamp]]
                                unique_verb_noun = list(set(zip(video_record._series['verb_class'],video_record._series['noun_class'])))
                                if (ek_ann["verb"],ek_ann["noun"]) not in unique_verb_noun:
                                    video_record._series['verb_class'][len(unique_verb_noun)] = ek_ann['verb_class'][0]
                                    video_record._series['noun_class'][len(unique_verb_noun)] = ek_ann['noun_class'][0]
                                    video_record._series['verb'][len(unique_verb_noun)] = ek_ann['verb'][0]
                                    video_record._series['noun'][len(unique_verb_noun)] = ek_ann['noun'][0]
                                self._video_records[video_times[video_timestamp]] = video_record
                            continue
                        if win_stop_sec > action_stop_sec and self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
                            self._video_records.append(EpicKitchensVideoRecord((tup[0],ek_ann)))
                            self._video_records[-1].time_end = video_durs[ek_ann['video_id']]
                            self._spatial_temporal_idx.append(0)
                            break
                        win_stop_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                        win_start_sec = win_stop_sec - self.cfg.TEST.SLIDE.WIN_SIZE
                        win_start_sec = win_start_sec if win_start_sec > 0.0  else 0.0
                        self._video_records.append(EpicKitchensVideoRecord((tup[0],ek_ann)))
                        self._video_records[-1].time_end = video_durs[ek_ann['video_id']]
                        self._spatial_temporal_idx.append(0)
                        iii += 1

        '''
        #
        # (ugly) code to find out the total number
        # of sliding windows that fit in the dataset
        # (including no action labels)
        #

        total_points = 0
        action_points = 0
        test_videos = list(set(['_'.join(k.split('_')[:2]) for k in video_times]))
        for k in test_videos:
            print(k)
            vdur = video_durs[k]
            video_range = np.arange(0,vdur,self.cfg.TEST.SLIDE.HOP_SIZE)
            print(np.floor(video_durs[k]/self.cfg.TEST.SLIDE.HOP_SIZE))
            vtimes = [kk for kk in video_times.keys() if '_'.join(kk.split('_')[:2])==k]
            vtimes = [float(k.split('_')[-1]) for k in vtimes]
            noaction_frames = len([t for t in video_range if t not in vtimes])
            print(len(vtimes))
            print(len(video_range) - noaction_frames,len(vtimes))
            assert len(video_range) - noaction_frames == len(vtimes)
            total_points += len(video_range)
            action_points += len(vtimes)

        print('total points', total_points)
        print('action points', action_points)
        '''

        if self.cfg.TEST.SLIDE and not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
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
        if self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
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

        if self.cfg.MODEL.MODEL_NAME == 'TSM':
            frames_rgb = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)
            frames_flow = pack_flow_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)
        else:
            frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)

        
        if self.cfg.MODEL.MODEL_NAME == 'SlowFast':
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

            label = self._video_records[index].label
            frames = utils.pack_pathway_output(self.cfg, frames)
            metadata = self._video_records[index].metadata
            return frames, label, index, metadata
        elif self.cfg.MODEL.MODEL_NAME == 'TSM':
            input_frames = []
            for i, frames in enumerate([frames_rgb, frames_flow]):
                scale = min_scale/frames.shape[1]
                frames = [
                        cv2.resize(
                            img_array.numpy(),
                            (0,0),
                            fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                        )
                        for img_array in frames
                ]
                frames = np.concatenate(
                    [np.expand_dims(img_array, axis=0) for img_array in frames],
                    axis=0,
                )
                frames = torch.from_numpy(np.ascontiguousarray(frames))
                frames = torch.flip(frames,dims=[3]) if i ==0 else frames # from bgr to rgb
                frames = frames.float()
                frames = frames / 255.0
                frames = frames - torch.tensor(self.cfg.DATA.MEAN) if i==0 else frames - 0.5
                frames = frames / torch.tensor(self.cfg.DATA.STD) if i==0 else frames/0.226
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                frames = self.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )
                input_frames.append(frames)
            label = self._video_records[index].label
            metadata = self._video_records[index].metadata
            return input_frames, label, index, metadata
        elif self.cfg.MODEL.MODEL_NAME == 'Omnivore':
            scale = min_scale/frames.shape[1]
            frames = [
                    cv2.resize(
                        img_array.numpy(),
                        (0,0),
                        fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                    )
                    for img_array in frames
            ]
            frames = np.concatenate(
                [np.expand_dims(img_array, axis=0) for img_array in frames],
                axis=0,
            )
            frames = torch.from_numpy(np.ascontiguousarray(frames))
            frames = torch.flip(frames,dims=[3]) # from bgr to rgb
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )
            label = self._video_records[index].label
            metadata = self._video_records[index].metadata
            return frames, label, index, metadata


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
