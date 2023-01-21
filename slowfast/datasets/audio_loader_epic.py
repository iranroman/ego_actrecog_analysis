import random
import numpy as np
import torch
import librosa
from librosa import stft, filters


def get_start_end_idx(audio_size, clip_size, clip_idx, num_clips, start_sample=0):
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
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    end_idx = start_idx + clip_size - 1
    return start_sample + start_idx, start_sample + end_idx


def _ensure_fit(start_time, end_time, duration):
    if start_time + duration > end_time:
        start_time = max(0, end_time - duration)
        duration = end_time - start_time
    return start_time, duration


def pack_audio(cfg, audio_dataset, record, temporal_sample_index):
    audio_path = os.path.join(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR, 'audio',
        record.untrimmed_video_name)
    # using units of time instead :)
    start_time, end_time = get_start_end_idx(
        record.duration,
        cfg.AUDIO_DATA.CLIP_SECS,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        start_sample=record.start_time
    )
    start_time, duration = _ensure_fit(start_time, end_time, cfg.AUDIO_DATA.CLIP_SECS)

    y, sr = librosa.load(
        audio_path, 
        sr=cfg.AUDIO_DATA.SAMPLING_RATE,
        mono=False,
        offset=start_time, 
        duration=duration)
    
    # ensure mono
    if len(y) > 1:
        y = y.mean(axis=0, keepdims=True)

    spec = _log_specgram(
        cfg, y,
        window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
        step_size=cfg.AUDIO_DATA.HOP_LENGTH,
        sr=cfg.AUDIO_DATA.SAMPLING_RATE,
    )
    spec = _pad_num_frames(cfg, spec)
    spec = torch.tensor(spec).unsqueeze(0)
    return spec


def _log_specgram(cfg, audio, sr, window_size=10, step_size=5, eps=1e-6):
    nperseg = int(round(window_size * sr / 1e3))
    noverlap = int(round(step_size * sr / 1e3))

    # mel-spec
    spec = librosa.stft(
        audio, n_fft=2048,
        window='hann',
        hop_length=noverlap,
        win_length=nperseg,
        pad_mode='constant')
    mel_basis = filters.mel(
        sr=sr,
        n_fft=2048,
        n_mels=128,
        htk=True,
        norm=None)
    mel_spec = np.dot(mel_basis, np.abs(spec))

    # log-mel-spec
    log_mel_spec = np.log(mel_spec + eps)
    return log_mel_spec.T


def _pad_num_frames(cfg, spectrogram):
    num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
    if num_timesteps_to_pad > 0:
        spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
    return spectrogram
