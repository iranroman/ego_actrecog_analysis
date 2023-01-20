import torch
from torch import nn

from .model_av import MTCN_AV
from .model_lm import MTCN_LM

from ..build import MODEL_REGISTRY
from ..auditory_slow_fast import AuditorySlowFast
from ..slow_fast import SlowFast  # noqa


@MODEL_REGISTRY.register()
class MTCN(nn.Module):
    def __init__(self, cfg):
        # create config copy so encoders return embeddings instead
        cfg_enc = cfg.clone()
        cfg_enc.MODEL.NUM_CLASSES = None

        # video encoder
        self.video_encoder = SlowFast(cfg_enc)

        # audio encoder
        self.audio_encoder = None
        if cfg.MTCN.AUDIO:
            self.audio_encoder = AuditorySlowFast(cfg_enc)

        # audio-visual cross attention
        self.cross = MTCN_AV(
            num_class=cfg.MODEL.NUM_CLASSES,
            visual_input_dim=cfg.RESNET.WIDTH_PER_GROUP * 32,
            audio_input_dim=cfg.RESNET.WIDTH_PER_GROUP * 32,
            seq_len=cfg.MTCN.SEQ_LEN,
            num_layers=cfg.MTCN.NUM_LAYERS,
            audio=cfg.MTCN.AUDIO)

    def forward(self, inputs):
        video, audio = inputs

        # extract video+audio features
        Zv = self.video_encoder(video)
        if self.audio_encoder is not None:
            Za = self.audio_encoder(audio)
            Z = torch.stack([Zv, Za], dim=1)
        else:
            Z = Zv[:, None]

        # cross-attention
        Y = self.cross(Z)
        return Y
