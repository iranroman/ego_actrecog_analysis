#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

_C.BN.FREEZE = False

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.ENABLE = True

# Dataset for testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.CHECKPOINT_TYPE = "pytorch"

_C.TEST.WEIGHT_INSTANCE_BY_NFRAMES = False

_C.TEST.SLIDE = CfgNode()
_C.TEST.SLIDE.ENABLE = False
_C.TEST.SLIDE.WIN_SIZE = 1.
_C.TEST.SLIDE.HOP_SIZE = 1.
_C.TEST.SLIDE.LABEL_FRAME = 0.5
_C.TEST.SLIDE.INSIDE_ACTION_BOUNDS = True
_C.TEST.SLIDE.PER_ACTION_INSTANCE = True

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.ARCH = "slowfast"

# Model name
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.NUM_CLASSES = [400, ]

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slowonly"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.FC_INIT_STD = 0.01

###############
# TSM configs #
###############
_C.MODEL.BASE_MODEL = 'resnet50'
_C.MODEL.NUM_SEGMENTS = 8
_C.MODEL.SEGMENT_LENGTH = [1,5] # RGB:1, Flow: 5
_C.MODEL.BEFORE_SOFTMAX = True
_C.MODEL.CROP_NUM = 1
_C.MODEL.CONCENSUS_TYPE = 'avg'
_C.MODEL.PARTIAL_BN = True
_C.MODEL.PRETRAINED = 'kinetics'
_C.MODEL.IS_SHIFT = True
_C.MODEL.SHIFT_DIV = 8
_C.MODEL.SHIFT_PLACE = 'blockres'
_C.MODEL.FC_LR5 = False
_C.MODEL.TEMPORAL_POOL = False
_C.MODEL.NON_LOCAL = False

# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# List of input frame channel dimensions.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.TEST_CROP_SIZE = 256

_C.DATA.FRAME_SAMPLING = 'like slowfast'


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NUM_GPUS = 1 

# Number of machine to use for the job.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NUM_SHARDS = 1

# The index of the current machine.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SHARD_ID = 0

# Output basedir.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

_C.EPICKITCHENS.VISUAL_DATA_DIR = ""

_C.EPICKITCHENS.VIDEO_DURS = "EPIC_100_video_info.csv"

#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.EPICKITCHENS.ANNOTATIONS_DIR = ""

_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_SPLIT = "validation"

_C.EPICKITCHENS.TRAIN_PLUS_VAL = False


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
