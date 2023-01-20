from torch import nn

from .model_av import MTCN_AV
from .model_lm import MTCN_LM

from ..build import MODEL_REGISTRY
from ..auditory_slow_fast import SlowFast as AuditorySlowFast
from ..slow_fast import SlowFast  # noqa


@MODEL_REGISTRY.register()
class MTCN(nn.Module):
    def __init__(self, cfg):
        self.video_encoder = SlowFast(cfg)
        self.audio_encoder = AuditorySlowFast(cfg)
        self.transformer = MTCN_AV(
            num_class=cfg.MODEL.NUM_CLASSES,
            seq_len=cfg.MODEL.SEQ_LEN,
            num_layers=cfg.MODEL.NUM_LAYERS)

    def forward(self, inputs):
        video, audio = inputs
        Zv = self.video_encoder(video)
        Za = self.audio_encoder(audio)
        Y = self.transformer([Zv, Za])
        return Y
