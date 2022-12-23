# modified from: 
# github.com/epic-kitchens/epic-kitchens-slowfast/blob/bf505199eb7d0b68adf2c8dcd847bc5b73949642/slowfast/datasets/epickitchens.py
import os
import pandas as pd
import torch
import torch.utils.data

#from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord

from . import transform as transform
#from . import utils as utils
from .frame_loader import pack_frames_to_video_clip

#logger = logging.get_logger(__name__)


class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, model=None):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        assert model in [
            None,
            "omnivore",
        ], "Model '{}' not supported for this dataset".format(model)
        self.cfg = cfg
        self.mode = mode
        self.model = model
        self.target_fps = 60
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_SPATIAL_CROPS
            )
        if model == 'omnivore':
            self.action2index, self.verb2index, self.noun2index = self._load_omnivore_indices()

        #logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()
        self.index2verb, self.index2noun = self._get_index_dicts()

    def _load_omnivore_indices(self):
        with open('model_metadata/epic_action_classes.csv') as f:
            data = f.read().splitlines()
        action2index = {d:i for i,d in enumerate(data)}

        verb_noun = pd.read_csv('model_metadata/epic_action_classes.csv', names=['verb','noun'])
        verbs = verb_noun['verb'].unique()
        nouns = verb_noun['noun'].unique()
        
        verb2index = {}
        for verb in verbs:
            verb2index[verb] = [v for k,v in action2index.items() if k.split(',')[0]==verb]
        noun2index = {}
        for noun in nouns:
            noun2index[noun] = [v for k,v in action2index.items() if k.split(',')[1]==noun]

        return action2index, verb2index, noun2index


    def _get_index_dicts(self):
        
        noun_classes = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, 'EPIC_100_noun_classes.csv'))
        verb_classes = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, 'EPIC_100_verb_classes.csv'))

        index2verb = pd.Series(verb_classes.key.values, index=verb_classes.id).to_dict()
        index2noun = pd.Series(noun_classes.key.values, index=noun_classes.id).to_dict()

        return index2verb, index2noun

    def _label2verbnounpair(self, label):
        verb = self.index2verb[label['verb']]
        noun = self.index2noun[label['noun']]
        return f'{verb},{noun}'

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            #path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
            # "testing" on the validation set due to lack of labels on the actual test set
        else:
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._video_records.append(EpicKitchensVideoRecord(tup))
                    self._spatial_temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        #logger.info(
        #    "Constructing epickitchens dataloader (size: {}) from {}".format(
        #        len(self._video_records), path_annotations_pickle
        #    )
        #)

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
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)
        
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
        
        #metadata = self._video_records[index].metadata
        if self.model:
            if self.model == 'omnivore':
                action_label = self._label2verbnounpair(self._video_records[index].label)
                #verb_label, noun_label = action_label.split(',')
                action_index = self.action2index[action_label]
                #verb_index = self.verb2index[verb_label]
                #noun_index = self.noun2index[noun_label]
            return frames, action_index#, verb_index, noun_index, metadata
        else:
            label = self._video_records[index].label
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
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
