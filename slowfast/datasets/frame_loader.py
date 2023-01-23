import os
import torch
import numpy as np
from . import utils as utils
from .decoder import get_start_end_idx


def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def omnivore_sampling(record, num_frames):
    seg_size = float(record.num_frames - 1) / num_frames
    i = np.arange(num_frames)
    return (
        record.start_frame + 
        np.round(seg_size * (i + 0.5)).astype(int)
    )


def pack_frames_to_video_clip(cfg, video_record, temporal_sample_index, target_fps=60):
    # Load video by loading its extracted frames
    if cfg.MODEL.MODEL_NAME == 'Omnivore':
        frame_idx = omnivore_sampling(video_record, cfg.DATA.NUM_FRAMES)
    else:
        fps, sr, num_samples = video_record.fps, cfg.DATA.SAMPLING_RATE, cfg.DATA.NUM_FRAMES
        start_idx, end_idx = get_start_end_idx(
            video_record.num_frames,
            num_samples * sr * fps / target_fps,
            temporal_sample_index,
            cfg.TEST.NUM_ENSEMBLE_VIEWS)
        frame_idx = temporal_sampling(
            video_record.num_frames,
            start_idx + 1, end_idx + 1, num_samples,
            start_frame=video_record.start_frame)

    return utils.retry_load_images([
        os.path.join(
            cfg.EPICKITCHENS.VISUAL_DATA_DIR, 
            'rgb_frames', 
            video_record.untrimmed_video_name,
            f"frame_{idx.item():010d}.jpg") 
        for idx in frame_idx
    ])
