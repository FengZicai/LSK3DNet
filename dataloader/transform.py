import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping


class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = int(self.sample_rate * data_dict["point_feat"].shape[0]) \
            if self.sample_rate is not None else self.point_max

        assert "point_feat" in data_dict.keys()
        
        if data_dict["point_feat"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["point_feat"][np.random.randint(data_dict["point_feat"].shape[0])]
            elif self.mode == "center":
                center = data_dict["point_feat"][data_dict["point_feat"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["point_feat"][:,:3] - center[:3]), 1))[:point_max]
            data_dict["point_feat"] = data_dict["point_feat"][idx_crop]
            data_dict["point_label"] = data_dict["point_label"][idx_crop]
            data_dict["normal"] = data_dict["normal"][idx_crop]
            data_dict["ref_index"] = data_dict["ref_index"][idx_crop]
            data_dict["point_num"] = data_dict["point_feat"].shape[0]

        return data_dict

class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "point_feat" in data_dict.keys():
            x_min, y_min, z_min = data_dict["point_feat"][:,:3].min(axis=0)
            x_max, y_max, _ = data_dict["point_feat"][:,:3].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["point_feat"][:,:3] -= shift
        return data_dict


class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "point_feat" in data_dict.keys()
        shuffle_index = np.arange(data_dict["point_feat"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict["point_feat"] = data_dict["point_feat"][shuffle_index]
        data_dict["point_label"] = data_dict["point_label"][shuffle_index]
        data_dict["normal"] = data_dict["normal"][shuffle_index]
        data_dict["ref_index"] = data_dict["ref_index"][shuffle_index]
        data_dict["point_num"] = data_dict["point_feat"].shape[0]

        return data_dict

class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []

        if self.cfg['SphereCrop']:
            self.transforms.append(SphereCrop(point_max=self.cfg['d_point_num'], mode="random"))
        if self.cfg['ShufflePoint']:
            self.transforms.append(ShufflePoint())

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict