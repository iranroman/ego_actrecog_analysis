# modified 
# github.com/epic-kitchens/epic-kitchens-slowfast/blob/bf505199eb7d0b68adf2c8dcd847bc5b73949642/slowfast/datasets/frame_loader.py
import os
import torch
import numpy as np
import time
import torch
import warnings
import cv2
import random

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        #start_idx = delta * clip_idx / num_clips
        start_idx = random.uniform(0, delta) # sample only one time.
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            print(image_paths)
            warnings.warn("Reading failed. Will retry.")
            time.sleep(1.0)
            input()
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

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


def pack_frames_to_video_clip(cfg, video_record, temporal_sample_index, target_fps=60):
    # Load video by loading its extracted frames
    path_to_video = '{}/rgb_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
    #                                             video_record.participant,
                                                 video_record.untrimmed_video_name)
    img_tmpl = "frame_{:010d}.jpg"
    fps, sampling_rate, num_samples = video_record.fps, cfg.DATA.SAMPLING_RATE, cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(video_record.num_frames,
                                  start_idx, end_idx, num_samples,
                                  start_frame=video_record.start_frame)
    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    frames = retry_load_images(img_paths)
    return frames
