#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
#from .video_model_builder import ResNet, SlowFast  # noqa

from .omnivore import Omnivore
from .auditory_slow_fast import AuditorySlowFast
from .slow_fast import SlowFast
from .mtcn import MTCN