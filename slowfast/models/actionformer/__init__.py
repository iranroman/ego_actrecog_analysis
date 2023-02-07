import torch
from torch import nn
from .blocks import (MaskedConv1D, MaskedMHCA, MaskedMHA, LayerNorm,
	                 TransformerBlock, ConvBlock, Scale, AffineDropPath)
from .models import make_backbone, make_neck, make_meta_arch, make_generator
from . import backbones      # backbones
from . import necks          # necks
from . import loc_generators # location generators
from . import meta_archs     # full models
from ..build import MODEL_REGISTRY
from ..slow_fast import SlowFast  # noqa

# __all__ = ['MaskedConv1D', 'MaskedMHCA', 'MaskedMHA', 'LayerNorm', 
#            'TransformerBlock', 'ConvBlock', 'Scale', 'AffineDropPath',
#            'make_backbone', 'make_neck', 'make_meta_arch', 'make_generator']



@MODEL_REGISTRY.register()
class Actionformer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.slowfast = SlowFast(cfg, return_features=True)
        self.actionformer = make_meta_arch(cfg.MODEL.META_ARCH)
        self.state_dict_ema = self.model # for weight loading
        self.fps = ...

    def forward(self, x, metadata):
        '''
        video_id, fps, duration, feat_stride, feat_num_frames
        '''
        y, features = self.slowfast(x)
        video_list = [
            {'feats': f, 'video_id': i, **m} 
            for i, (f, m) in enumerate(zip(features, metadata))
        ]
        output = self.actionformer(video_list)
        t0, t1, label, score = unpack_results(output)
        return t0, t1, label, score


def unpack_results(output):
    # unpack the results into ANet format
    results = {k: [] for k in ['t-start', 't-end', 'label', 'score', 'video-id']}
    for vid_idx, o in enumerate(output):
        if o['segments'].shape[0]:
            # results['video-id'].extend([o['video_id']] * o['segments'].shape[0])
            results['t-start'].append(o['segments'][:, 0])
            results['t-end'].append(o['segments'][:, 1])
            results['label'].append(o['labels'])
            results['score'].append(o['scores'])
    return [torch.as_tensor(results[k]) for k in ['t-start', 't-end', 'label', 'score']]


