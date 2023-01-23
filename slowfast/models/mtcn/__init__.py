import torch
from torch import nn

from .model_av import MTCN_AV
from .model_lm import MTCN_LM

from ..build import MODEL_REGISTRY
from ..auditory_slow_fast import AuditorySlowFast
from ..slow_fast import SlowFast  # noqa


@MODEL_REGISTRY.register
class MTCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # create config copy so encoders return embeddings instead
        cfg2 = cfg.clone()
        cfg2.MODEL.NUM_CLASSES = None

        # video encoder
        self.video_encoder = SlowFast(cfg2)

        # audio encoder
        self.audio_encoder = None
        if cfg.MODEL.AUDIO:
            self.audio_encoder = AuditorySlowFast(cfg2)

        # audio-visual cross attention
        self.cross_attention = MTCN_AV(
            num_class=cfg.MODEL.NUM_CLASSES,
            visual_input_dim=cfg.RESNET.WIDTH_PER_GROUP * 32,
            audio_input_dim=cfg.RESNET.WIDTH_PER_GROUP * 32,
            seq_len=cfg.MTCN.SEQ_LEN,
            num_layers=cfg.MTCN.NUM_LAYERS,
            audio=cfg.MODEL.AUDIO)

    def forward(self, inputs):
        # extract video+audio features
        if self.audio_encoder is not None:
            video, audio = inputs
            Zv = self.video_encoder(video)
            Za = self.audio_encoder(audio)
            Z = torch.stack([Zv, Za], dim=1)
        # video only
        else:
            Zv = self.video_encoder(inputs)
            Z = Zv[:, None]

        # cross-attention
        Y = self.cross_attention(Z)
        return Y
