#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import re
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo
from copy import deepcopy

import slowfast.utils.weight_init_helper as init_helper

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from .temporal_shift import make_temporal_shift
from .basic_ops import ConsensusModule

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    #"c2d": [
    #    [[1]],  # conv1 temporal kernel.
    #    [[1]],  # res2 temporal kernel.
    #    [[1]],  # res3 temporal kernel.
    #    [[1]],  # res4 temporal kernel.
    #    [[1]],  # res5 temporal kernel.
    #],
    #"c2d_nopool": [
    #    [[1]],  # conv1 temporal kernel.
    #    [[1]],  # res2 temporal kernel.
    #    [[1]],  # res3 temporal kernel.
    #    [[1]],  # res4 temporal kernel.
    #    [[1]],  # res5 temporal kernel.
    #],
    #"i3d": [
    #    [[5]],  # conv1 temporal kernel.
    #    [[3]],  # res2 temporal kernel.
    #    [[3, 1]],  # res3 temporal kernel.
    #    [[3, 1]],  # res4 temporal kernel.
    #    [[1, 3]],  # res5 temporal kernel.
    #],
    #"i3d_nopool": [
    #    [[5]],  # conv1 temporal kernel.
    #    [[3]],  # res2 temporal kernel.
    #    [[3, 1]],  # res3 temporal kernel.
    #    [[3, 1]],  # res4 temporal kernel.
    #    [[1, 3]],  # res5 temporal kernel.
    #],
    #"slowonly": [
    #    [[1]],  # conv1 temporal kernel.
    #    [[1]],  # res2 temporal kernel.
    #    [[1]],  # res3 temporal kernel.
    #    [[3]],  # res4 temporal kernel.
    #    [[3]],  # res5 temporal kernel.
    #],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    #"c2d": [[2, 1, 1]],
    #"c2d_nopool": [[1, 1, 1]],
    #"i3d": [[2, 1, 1]],
    #"i3d_nopool": [[1, 1, 1]],
    #"slowonly": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.enable_detection = False
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'bn_parameters':
            print("Freezing all BN layers\' parameters.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == 'bn_statistics':
            print("Freezing all BN layers\' statistics.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown running statistics update in frozen mode
                    m.eval()


@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.omni = torch.hub.load("facebookresearch/omnivore:main", model=cfg.MODEL.ARCH)
        self.register_buffer('verb_matrix',self._get_output_transform_matrix('verb',cfg))
        self.register_buffer('noun_matrix',self._get_output_transform_matrix('noun',cfg))

    def _get_output_transform_matrix(self, which_one,cfg):

        with open('slowfast/models/omnivore_epic_action_classes.csv') as f:
            data = f.read().splitlines()
            action2index = {d:i for i,d in enumerate(data)}


        if which_one == 'verb':
            verb_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_verb_classes.csv',usecols=['key'])
            verb2index = {}
            for verb in verb_classes['key']:
                verb2index[verb] = [v for k,v in action2index.items() if k.split(',')[0]==verb]
            matrix = torch.zeros(len(action2index),len(verb2index))
            for i, (k,v) in enumerate(verb2index.items()):
                for j in v:
                    matrix[j,i] = 1.
        elif which_one == 'noun':
            noun_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_noun_classes.csv',usecols=['key'])
            noun2index = {}
            for noun in noun_classes['key']:
                noun2index[noun] = [v for k,v in action2index.items() if k.split(',')[1]==noun]
            matrix = torch.zeros(len(action2index),len(noun2index))
            for i, (k,v) in enumerate(noun2index.items()):
                for j in v:
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


def strip_module_prefix(state_dict):
    return {re.sub("^module.", "", k): v for k, v in state_dict.items()}


@MODEL_REGISTRY.register()
class TSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_segments = cfg.MODEL.NUM_SEGMENTS
        self.segment_length = cfg.MODEL.SEGMENT_LENGTH
        self.reshape = True
        self.before_softmax = cfg.MODEL.BEFORE_SOFTMAX
        self.dropout = cfg.MODEL.DROPOUT_RATE
        self.crop_num = cfg.MODEL.CROP_NUM
        self.consensus_type = cfg.MODEL.CONCENSUS_TYPE
        self.is_shift = cfg.MODEL.IS_SHIFT
        self.shift_div = cfg.MODEL.SHIFT_DIV
        self.shift_place = cfg.MODEL.SHIFT_PLACE
        self.base_model_name = cfg.MODEL.BASE_MODEL
        self.fc_lr5 = cfg.MODEL.FC_LR5
        self.temporal_pool = cfg.MODEL.TEMPORAL_POOL
        self.non_local = cfg.MODEL.NON_LOCAL
        self.num_classes = cfg.MODEL.NUM_CLASSES

        if not cfg.MODEL.BEFORE_SOFTMAX and consensus_type != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        print(
            f"""
    Initializing {self.__class__.__name__} with base model: {self.base_model_name}.

    {self.__class__.__name__} Configuration:
        num_segments:       {self.num_segments}
        segment_length:     {self.segment_length}
        consensus_module:   {self.consensus_type}
        dropout_ratio:      {self.dropout}
            """
        )

        self.base_model_rgb = self._prepare_base_model(self.base_model_name)

        self.base_model_rgb = self._prepare_tsn(self.base_model_rgb, cfg.MODEL.NUM_CLASSES)

        print("Creating model that operates on optical flow")
        self.base_model_flow = self._construct_flow_model(self.base_model_rgb)

        self.consensus = ConsensusModule(self.consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = cfg.MODEL.PARTIAL_BN
        if cfg.MODEL.PARTIAL_BN:
            self.partialBN(True)

    def _prepare_tsn(self, base_model, num_class):
        feature_dim = getattr(
            base_model, base_model.last_layer_name
        ).in_features
        if self.dropout == 0:
            setattr(
                base_model,
                base_model.last_layer_name,
                nn.Linear(feature_dim, sum(num_class)),
            )
            self.new_fc = None
        else:
            setattr(
                base_model,
                base_model.last_layer_name,
                nn.Dropout(p=self.dropout),
            )
            base_model.new_fc = nn.Linear(feature_dim, sum(num_class))

        std = 0.001
        if base_model.new_fc is None:
            normal_(
                getattr(base_model, base_model.last_layer_name).weight, 0, std
            )
            constant_(getattr(base_model, base_model.last_layer_name).bias, 0)
        else:
            if hasattr(base_model.new_fc, "weight"):
                normal_(base_model.new_fc.weight, 0, std)
                constant_(base_model.new_fc.bias, 0)
        return base_model

    def _prepare_base_model(self, base_model):
        print(f"base model: {base_model}")

        if "resnet" in base_model:
            _model = getattr(torchvision.models, base_model)(
                pretrained=None
            )
            if self.is_shift:
                print("Adding temporal shift...")

                make_temporal_shift(
                    _model,
                    self.num_segments,
                    n_div=self.shift_div,
                    place=self.shift_place,
                    temporal_pool=self.temporal_pool,
                )

            if self.non_local:
                print("Adding non-local module...")
                from ..ops.non_local import make_non_local

                make_non_local(self.base_model, self.num_segments)

            _model.last_layer_name = "fc"

            _model.avgpool = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError(f"Unknown base model: {base_model!r}")
        return _model

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSM, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if (
                isinstance(m, torch.nn.Conv2d)
                or isinstance(m, torch.nn.Conv1d)
                or isinstance(m, torch.nn.Conv3d)
            ):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".format(
                            type(m)
                        )
                    )

        return [
            {
                "params": first_conv_weight,
                "lr_mult": 5 if self.modality == "Flow" else 1,
                "decay_mult": 1,
                "name": "first_conv_weight",
            },
            {
                "params": first_conv_bias,
                "lr_mult": 10 if self.modality == "Flow" else 2,
                "decay_mult": 0,
                "name": "first_conv_bias",
            },
            {
                "params": normal_weight,
                "lr_mult": 1,
                "decay_mult": 1,
                "name": "normal_weight",
            },
            {
                "params": normal_bias,
                "lr_mult": 2,
                "decay_mult": 0,
                "name": "normal_bias",
            },
            {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "BN scale/shift"},
            {"params": custom_ops, "lr_mult": 1, "decay_mult": 1, "name": "custom_ops"},
            # for fc
            {"params": lr5_weight, "lr_mult": 5, "decay_mult": 1, "name": "lr5_weight"},
            {"params": lr10_bias, "lr_mult": 10, "decay_mult": 0, "name": "lr10_bias"},
        ]

    def forward(self, x_y, no_reshape=False):
        x, y = x_y
        x = torch.permute(x, (0,2,1,3,4))
        y = torch.permute(y, (0,2,1,3,4))
        nchans_rgb = x.size()[2]
        nchans_flow = y.size()[2]
        base_out_rgb = self.base_model_rgb(x.reshape((-1,nchans_rgb)+x.size()[-2:]))
        base_out_flow = self.base_model_flow(y.reshape((-1,nchans_flow)+y.size()[-2:]))

        if self.dropout > 0:
            base_out_rgb = self.base_model_rgb.new_fc(base_out_rgb)
            base_out_flow = self.base_model_flow.new_fc(base_out_flow)

        if not self.before_softmax:
            base_out_rgb = self.softmax(base_out_rgb)
            base_out_flow = self.softmax(base_out_flow)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view(
                    (-1, self.num_segments // 2) + base_out.size()[1:]
                )
            else:
                base_out_rgb = base_out_rgb.view((-1, self.num_segments) + base_out_rgb.size()[1:])
                base_out_flow = base_out_flow.view((-1, self.num_segments) + base_out_flow.size()[1:])
            output_rgb = self.consensus(base_out_rgb)
            output_flow = self.consensus(base_out_flow)
            output = (output_rgb + output_flow) / 2
            output = output.squeeze(1)
            return output[:,:self.num_classes[0]], output[:,self.num_classes[0]:]

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(
            filter(
                lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
            )
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        container = deepcopy(container)

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.segment_length[1],) + kernel_size[2:] # the second element in segment is the one for Flow
        new_kernels = (
            params[0]
            .data.mean(dim=1, keepdim=True)
            .expand(new_kernel_size)
            .contiguous()
        )

        new_conv = nn.Conv2d(
            2 * self.segment_length[1], # the second element in segment is the one for Flow
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return container
