'''
XXX: There is a small bug in here. Use audio_loader_epic_og.py for paper numbers until we figure it out.
'''
from typing import Literal
import os
import random
import numpy as np
import torch
import librosa


def get_start_end(region_size, clip_size, clip_idx, num_clips=None, hop_size=None, offset=0):
    """
    Sample a clip of size clip_size from an audio of size audio_size and
    return the indices of the first and last sample of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the audio to
    num_clips clips, and select the start and end index of clip_idx-th audio
    clip.
    Args:
        region_size (int): number of overall seconds in recording segment.
        clip_size (int): size of the clip to sample from the samples.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the audio to num_clips
            clips, and select the start and end index of the clip_idx-th audio
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given audio for testing.
    Returns:
        start_idx (int): the start sample index.
        end_idx (int): the end sample index.
    """
    delta = max(region_size - clip_size, 0)
    if hop_size:
        start_idx = np.arange(0, delta, hop_size)[clip_idx]
    elif num_clips:  # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    else:#if clip_idx == -1:  # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    return offset + start_idx, offset + start_idx + clip_size






def pack_audio(cfg, record, clip_index, num_clips, audio_dataset=None):
    hop_size = None
    clip_duration = cfg.AUDIO_DATA.CLIP_SECS
    if cfg.TEST.SLIDE.ENABLE:
        clip_duration = cfg.TEST.SLIDE.WIN_SIZE
        hop_size = cfg.TEST.SLIDE.HOP_SIZE
    start_time, end_time = record.start_time, record.end_time
    if record.duration > clip_duration:
        # get the start and end time of the clip
        # print(start_time, end_time)
        start_time, end_time = get_start_end(
            record.duration, clip_duration,
            clip_index, num_clips, 
            hop_size=hop_size,
            offset=record.start_time)
        end_time -= 1/cfg.AUDIO_DATA.SAMPLING_RATE
    # print(start_time, end_time, record.duration, clip_duration)
    

    y, sr = _load_audio(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR, 
        record.untrimmed_video_name, 
        cfg.AUDIO_DATA.SAMPLING_RATE,
        start_time, end_time, audio_dataset=audio_dataset)

    # extract audio features
    spec = _log_specgram(
        y, sr,
        win_length=cfg.AUDIO_DATA.WINDOW_LENGTH,
        hop_length=cfg.AUDIO_DATA.HOP_LENGTH,
        n_mels=cfg.AUDIO_DATA.NUM_FREQUENCIES)
    spec = _pad_num_frames(cfg, spec)
    spec = torch.tensor(spec).unsqueeze(0)
    return spec


def _load_audio(ek_dir, video_name, sr, start_time, end_time, audio_dataset=None):
    # load the audio clip
    if audio_dataset is None:
        audio_path = os.path.join(ek_dir, 'audio', f'{video_name}.wav')
        y, sr = librosa.load(
            audio_path, sr=sr, mono=False, 
            offset=start_time, duration=end_time - start_time)
    else:
        y = audio_dataset[video_name][int(start_time * sr):int(end_time * sr)]
 

    if y.ndim > 1:  # convert to mono
        assert len(y) <= 2, f"Expected mono or stereo audio. Check loaded audio dimension order {y.shape}."
        y = y.mean(axis=0)
    return y, sr


def _log_specgram(audio, sr, win_length=10, hop_length=5, n_mels=128, n_fft=2048, eps=1e-6):
    spec = librosa.stft(
        audio, n_fft=n_fft,
        window='hann',
        win_length=int(round(win_length * sr / 1000.)),
        hop_length=int(round(hop_length * sr / 1000.)),
        pad_mode='constant')

    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, 
        htk=True, norm=None)

    mel_spec = np.dot(mel_basis, np.abs(spec))
    log_mel_spec = np.log(mel_spec + eps)
    return log_mel_spec.T


def _pad_num_frames(cfg, S):
    padlen = cfg.AUDIO_DATA.NUM_FRAMES - len(S)
    return np.pad(S, ((0, padlen), (0, 0)), 'edge') if padlen > 0 else S
