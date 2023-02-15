import os
import torch
import numpy as np
from . import utils as utils
from .decoder import get_start_end_idx


def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0, video_last_frame=float('inf'), action_last_frame = float('inf')):
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
    index += start_frame
    end_frame = start_frame + num_frames - 1
    index = torch.clamp(index, start_frame, end_frame if (end_frame < action_last_frame and end_frame < video_last_frame) else video_last_frame if video_last_frame < action_last_frame else action_last_frame).long()
    return index


def pack_frames_to_video_clip(cfg, video_record, temporal_sample_index, target_fps=60):
    # Load video by loading its extracted frames
    img_tmpl = "frame_{:010d}.jpg"
    if cfg.TEST.SLIDE.ENABLE:
        start_idx = 0
        end_idx = cfg.TEST.SLIDE.WIN_SIZE * video_record.fps - 1
        frame_idx = temporal_sampling(video_record.num_frames+1,
                                      start_idx, end_idx, cfg.DATA.NUM_FRAMES,
                                      start_frame=video_record.start_frame + 1, 
                                      video_last_frame=video_record.time_end*video_record.fps,
                                      action_last_frame=video_record._series['action_stop_frame'] if (cfg.TEST.SLIDE.ENABLE and cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS) else float('inf'))
    elif cfg.DATA.FRAME_SAMPLING == 'like slowfast':
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
    elif cfg.DATA.FRAME_SAMPLING == 'like omnivore':

        seg_size = float(video_record.num_frames - 1) / cfg.DATA.NUM_FRAMES
        seq = []
        for i in range(cfg.DATA.NUM_FRAMES):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)
        frame_idx = torch.tensor(video_record.start_frame + np.array(seq))
    path_to_video = '{}/rgb_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                 #video_record.participant,
                                                 video_record.untrimmed_video_name)
    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths)
    return frames

def pack_flow_frames_to_video_clip(cfg, video_record, temporal_sample_index, target_fps=60):
    # Load video by loading its extracted frames
    img_tmpl = "frame_{:010d}.jpg"
    if cfg.TEST.SLIDE.ENABLE:
        start_idx = 0
        end_idx = cfg.TEST.SLIDE.WIN_SIZE * int(video_record.fps/2) - 1 # half of frames for flow compared to RGB
        frame_idx = temporal_sampling(int(video_record.num_frames/2)+1,
                                      start_idx, end_idx, cfg.DATA.NUM_FRAMES,
                                      start_frame=int(video_record.start_frame//2) + 1, 
                                      video_last_frame=np.floor(video_record.time_end*video_record.fps)/2,
                                      action_last_frame=int(video_record._series['action_stop_frame']//2) if (cfg.TEST.SLIDE.ENABLE and cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS) else float('inf'))
    elif cfg.DATA.FRAME_SAMPLING == 'like slowfast':
        fps, sampling_rate, num_samples = video_record.fps, cfg.DATA.SAMPLING_RATE, cfg.DATA.NUM_FRAMES
        start_idx, end_idx = get_start_end_idx(
            video_record.num_frames,
            num_samples * sampling_rate * fps / target_fps,
            temporal_sample_index,
            cfg.TEST.NUM_ENSEMBLE_VIEWS,
        )
        start_idx, end_idx = int(start_idx/2) + 1, int(end_idx/2) + 1
        frame_idx = temporal_sampling(int(video_record.num_frames/2),
                                      start_idx, end_idx, num_samples,
                                      start_frame=int(video_record.start_frame/2))
    elif cfg.DATA.FRAME_SAMPLING == 'like omnivore':

        seg_size = float(int(video_record.num_frames/2) - 1) / cfg.DATA.NUM_FRAMES
        seq = []
        for i in range(cfg.DATA.NUM_FRAMES):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)
        frame_idx = torch.tensor(int(video_record.start_frame/2) + np.array(seq))
    frame_idx = torch.repeat_interleave(frame_idx,cfg.MODEL.SEGMENT_LENGTH[1]) + torch.arange(cfg.MODEL.SEGMENT_LENGTH[1]).repeat(cfg.DATA.NUM_FRAMES)
    frame_idx = torch.clamp(frame_idx,max=int(video_record._series['action_stop_frame']//2) if (cfg.TEST.SLIDE.ENABLE and cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS) else (np.floor(video_record.time_end*video_record.fps/2))).long()
    path_to_video = '{}/flow/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                 #video_record.participant,
                                                 video_record.untrimmed_video_name)
    img_paths_u = [os.path.join(path_to_video, 'u',img_tmpl.format(idx.item())) for idx in frame_idx]
    img_paths_v = [os.path.join(path_to_video, 'v', img_tmpl.format(idx.item())) for idx in frame_idx]
    img_paths = [val for pair in zip(img_paths_u, img_paths_v) for val in pair]
    frames = utils.retry_load_images(img_paths, flow=True)
    frames = torch.reshape(frames,(cfg.DATA.NUM_FRAMES, cfg.MODEL.SEGMENT_LENGTH[1]*2)+frames.size()[-2:])
    frames = torch.permute(frames,(0,2,3,1))
    return frames
