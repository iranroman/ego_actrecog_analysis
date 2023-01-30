import torch
from torch.utils import data
import pandas as pd
import numpy as np
import h5py


class EpicKitchens(data.Dataset):
    def __init__(self,
                 hdf5_path,
                 labels_pickle,
                 visual_feature_dim=2304,
                 audio_feature_dim=2304,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 labels_mode='center_action'):
        self.hdf5_dataset = None
        self.hdf5_path = hdf5_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.window_len = window_len
        self.num_clips = num_clips
        assert clips_mode in ['all', 'random'], \
            "Labels mode not supported. Choose from ['all', 'random']"
        assert labels_mode in ['all', 'center_action'], \
            "Labels mode not supported. Choose from ['all', 'center_action']"
        self.clips_mode = clips_mode
        self.labels_mode = labels_mode

    def __getitem__(self, index):
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File(self.hdf5_path, 'r')
        h5d = self.hdf5_dataset

        record = self.df_labels.iloc[index]
        narration_id = record.name
        video_id = record.video_id
    
        df_sorted_video = self.df_labels[self.df_labels.video_id == video_id].sort_values('start_timestamp')
        idx = df_sorted_video.index.get_loc(narration_id)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1

        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_narration_ids = df_sorted_video.iloc[sequence_range].index.tolist()

        num_clips = self.num_clips if self.clips_mode == 'all' else 1
        data = torch.zeros((
            2 * self.window_len * num_clips, 
            max(self.visual_feature_dim, self.audio_feature_dim)))
        if self.clips_mode == 'random':
            for i, ni in enumerate(sequence_narration_ids):
                self._set_clip(data, h5d, ni, i, np.random.randint(self.num_clips), num_clips)
        else:
            for i, ni in enumerate(sequence_narration_ids):
                for j in range(self.num_clips):
                    self._set_clip(data, h5d, ni, i, j, num_clips, j)

        idx = sequence_range if self.labels_mode == "all" else idx
        label = {
            'verb': self._get_label(df_sorted_video.verb_class.values[idx]),
            'noun': self._get_label(df_sorted_video.noun_class.values[idx]),
        }

        return data, label, narration_id

    def _set_clip(self, data, h5d, ni, i, j, num_clips, jj=None):
        i = i * num_clips + (j if jj is None else jj)
        data[i, :self.visual_feature_dim] = torch.from_numpy(h5d['visual_features'][ni][j])
        data[i + len(data) // 2, :self.audio_feature_dim] = torch.from_numpy(h5d['audio_features'][ni][j])

    def _get_label(self, classes):
        if classes is None:
            classes = torch.full((self.window_len if self.labels_mode == "all" else 1,), -1)
        if self.labels_mode == "all":
            classes = torch.from_numpy(classes).repeat(2)
            return torch.cat([classes, classes[self.window_len // 2].unsqueeze(0)])
        return torch.tensor(classes)


    def __len__(self):
        return self.df_labels.shape[0]