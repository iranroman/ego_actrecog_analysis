import os
import random
import numpy as np
import torch
import librosa
from librosa import stft, filters


def get_start_end(audio_size, clip_size, clip_idx, num_clips, offset=0):
    """
    Sample a clip of size clip_size from an audio of size audio_size and
    return the indices of the first and last sample of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the audio to
    num_clips clips, and select the start and end index of clip_idx-th audio
    clip.
    Args:
        audio_size (int): number of overall samples.
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
    delta = max(audio_size - clip_size, 0)
    # print(delta, audio_size, clip_size, clip_idx, num_clips, offset)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    end_idx = start_idx + clip_size - 1
    return offset + start_idx, offset + end_idx


def _ensure_fit(start_time, end_time, duration):
    '''If duration extends past the end of the clip, try to fit it 
    within the clip bounds.'''
    if start_time + duration > end_time:
        start_time = max(0, end_time - duration)
        duration = end_time - start_time
    return start_time, duration


def pack_audio(cfg, record, temporal_sample_index, num_clips, h5_reader=None):
    # get the start and end time of the clip
    start_time, end_time = get_start_end(
        record.duration,
        cfg.AUDIO_DATA.CLIP_SECS,
        temporal_sample_index,
        num_clips,
        offset=record.start_time)
    start_time, duration = _ensure_fit(start_time, end_time, cfg.AUDIO_DATA.CLIP_SECS)

    # load the audio clip
    if h5_reader is None:
        audio_path = os.path.join(
            cfg.EPICKITCHENS.VISUAL_DATA_DIR, 'audio',
            f'{record.untrimmed_video_name}.wav')
        y, sr = librosa.load(
            audio_path, 
            sr=cfg.AUDIO_DATA.SAMPLING_RATE,
            mono=False,
            offset=start_time, 
            duration=duration)
    else:
        sr = cfg.AUDIO_DATA.SAMPLING_RATE
        y = h5_reader[record.untrimmed_video_name][int(start_time * sr):int((start_time + duration) * sr)]
    # y = np.atleast_2d(y)
    
    # # check number of channels
    # channels = cfg.AUDIO_DATA.CHANNELS
    # if len(y) != channels:
    #     if channels != 1:
    #         raise RuntimeError(f"Unexpected number of channels. got {len(y)}, expected {channels}")
    #     # convert to mono
    #     y = y.mean(axis=0, keepdims=True)
    
    if y.ndim > 1:
        # convert to mono
        y = y.mean(axis=0)

    # extract audio features
    spec = _log_specgram(
        cfg, y,
        win_length=cfg.AUDIO_DATA.WINDOW_LENGTH,
        hop_length=cfg.AUDIO_DATA.HOP_LENGTH,
        n_mels=cfg.AUDIO_DATA.NUM_FREQUENCIES,
        sr=sr)
    spec = _pad_num_frames(cfg, spec)
    spec = torch.tensor(spec).unsqueeze(0)
    return spec

def _log_specgram(cfg, audio, sr, win_length=10, hop_length=5, n_mels=128, eps=1e-6):
    # log mel-spec
    spec = librosa.stft(
        audio, n_fft=2048,
        window='hann',
        win_length=int(round(win_length * sr / 1000.)),
        hop_length=int(round(hop_length * sr / 1000.)),
        pad_mode='constant')

    mel_basis = filters.mel(
        sr=sr, n_fft=2048, n_mels=n_mels, 
        htk=True, norm=None)

    mel_spec = np.dot(mel_basis, np.abs(spec))
    log_mel_spec = np.log(mel_spec + eps)
    return log_mel_spec.T


def _pad_num_frames(cfg, spectrogram):
    num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
    if num_timesteps_to_pad > 0:
        spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
    return spectrogram
