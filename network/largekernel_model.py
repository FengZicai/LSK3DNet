from torch import nn
import torch
import numpy as np
import copy
from network.voxel_fea_generator import voxel_3d_generator, voxelization
from network.voxel_fea_generator import voxel_3d_generator_fixvs, voxelization_fixvs

from network.spvcnn import SPVBlock
from network.spvcnn import SPVBlock_fixvs

import spconv.pytorch as spconv

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class largekernelseg(nn.Module):
    def __init__(self, config):
        super(largekernelseg, self).__init__()
        self.input_dims = config['model_params']['input_dims']
        self.hiden_size = config['model_params']['hiden_size']
        self.large_kernel = config['model_params']['large_kernel_size']
        self.num_classes = config['model_params']['num_classes']
        self.scale_list = config['model_params']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config['dataset_params']['min_volume_space']
        max_volume_space = config['dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config['model_params']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                large_kernel=self.large_kernel,
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist())
                # spatial_shape=np.int32(self.spatial_shape // self.strides[i]).tolist())
            )
        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # self.projection_layer = nn.Sequential(
        #         nn.Linear(self.hiden_size * self.num_scales, 256), # 232 
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(True),
        #         nn.Linear(256, 64),
        #         nn.ReLU(True),
        #     )

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))

        output = torch.cat(enc_feats, dim=1)
        data_dict['logits'] = self.classifier(output)
        # data_dict['cc_feature'] = self.projection_layer(output)

        return data_dict
    

@register_model
class largekernelseg_fixvs(nn.Module):
    def __init__(self, config):
        super(largekernelseg_fixvs, self).__init__()
        self.input_dims = config['model_params']['input_dims']
        self.hiden_size = config['model_params']['hiden_size']
        self.large_kernel = config['model_params']['large_kernel_size']
        self.num_classes = config['model_params']['num_classes']
        self.scale_list = config['model_params']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config['dataset_params']['min_volume_space']
        max_volume_space = config['dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config['model_params']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]
        self.voxel_size = config['dataset_params']['voxel_size']
        
        # voxelization
        self.voxelizer = voxelization_fixvs(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list,
            voxel_size=self.voxel_size
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator_fixvs(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            voxel_size=self.voxel_size
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock_fixvs(
                large_kernel=self.large_kernel,
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist())
                # spatial_shape=np.int32(self.spatial_shape // self.strides[i]).tolist())
            )
        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )


    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))

        output = torch.cat(enc_feats, dim=1)
        data_dict['logits'] = self.classifier(output)

        return data_dict
