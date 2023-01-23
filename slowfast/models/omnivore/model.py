import os
import torch
import torch.nn as nn
import pandas as pd

from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.omni = torch.hub.load("facebookresearch/omnivore:main", model=cfg.MODEL.ARCH)
        verb_matrix, noun_matrix = self._get_output_transform_matrix(cfg)
        self.register_buffer('verb_matrix', verb_matrix)
        self.register_buffer('noun_matrix', noun_matrix)

    def _get_output_transform_matrix(self, cfg):
        with open(os.path.join(os.path.dirname(__file__), 'omnivore_epic_action_classes.csv')) as f:
            action2index = {d.strip(): i for i, d in enumerate(f.readlines())}
        verb_matrix = self._construct_matrix(action2index, os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, 'EPIC_100_verb_classes.csv'), 0)
        noun_matrix = self._construct_matrix(action2index, os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, 'EPIC_100_noun_classes.csv'), 1)
        return verb_matrix, noun_matrix

    def _construct_matrix(self, action2index, fname, i_a):
        classes = pd.read_csv(fname, usecols=['id', 'key']).set_index('id').key
        matrix = torch.zeros(len(action2index), len(classes))
        for i, x in enumerate(classes):
            for a, j in action2index.items():
                if a.split(',')[i_a] == x:
                    matrix[j,i] = 1.
        return matrix

    def forward(self, x):
        y = self.omni(x, input_type="video")

        # must relocate the following to be able to train
        # using these matrices will also make it impossible to
        # get topk accuracies for k>1
        verb_noun_index = torch.argmax(y,dim=-1, keepdims=True)
        y_hardmax = torch.zeros_like(y).scatter_(1, verb_noun_index, 1.0)
        verb = y_hardmax @ self.verb_matrix
        noun = y_hardmax @ self.noun_matrix
        #verb, noun = self._omnioutput2verbnoun(y_hardmax) 
        return [verb, noun]
