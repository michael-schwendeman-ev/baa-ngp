"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from easydict import EasyDict as edict
import numpy as np
from pose_utils import invert_pose, to_hom, construct_pose, compose_poses
import torch


# basic operations of transforming 3D points between world/camera/image coordinates
def cam2world(X, pose_inv):
    """
    Args:
        X: 3D points in camera space with [H*W, 3]
        pose: camera pose. world_to_camera #camera_from_world, or world_to_camera [3, 3]
    Returns:
        transformation in world coordinate system.
    """
    # X of shape 64x3
    # X_hom is of shape 64x4
    X_hom = to_hom(X)
    # pose_inv is world_from_camera pose is of shape 3x4
    # pose_inv = invert_pose(pose)
    # world = camera * world_from_camera
    return X_hom@pose_inv.transpose(-1,-2)


def img2cam(X, cam_intr):
    """
    Args:
        X: 3D points in image space.
        cam_intr: camera intrinsics. cam_to_image? #camera_from_image
    Returns:
        trasnformation in camera coordinate system
    """
    # camera = image * image_from_camera
    return X@cam_intr.inverse().transpose(-1,-2)

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

