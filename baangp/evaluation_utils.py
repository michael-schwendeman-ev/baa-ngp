"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import torch.nn.functional as F
import tqdm
from utils import (
    render_image_with_occgrid,
)
from pose_utils import construct_pose
from camera_utils import cam2world, rotation_distance#, procrustes_analysis
from lie_utils import se3_to_SE3, so3_t3_to_SE3, so3_to_SE3
from easydict import EasyDict as edict
from utils import Rays



def evaluate_camera_alignment(pose_aligned,pose_GT):
    """
    Evaluate camera pose estimation with average Rotation and Translation errors.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L125
    """
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = rotation_distance(R_aligned, R_GT, 1e-15)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    error = edict(R=R_error, t=t_error)
    return error

