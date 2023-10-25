# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np
import sys
import typing
import torch

from .ngp import NGPRadianceField

sys.path.append("..")
from camera_utils import cam2world
from lie_utils import se3_to_SE3, so3_to_SE3, t3_to_SE3
from pose_utils import construct_pose, compose_poses
from utils import Rays


class BAradianceField(torch.nn.Module):
    """Bundle-Ajusting radiance field."""
    def __init__(
        self,
        num_frame: int,  # The number of frames.
        aabb: typing.Union[torch.Tensor, typing.List[float]],
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        n_features_per_level: int = 2, 
        log2_hashmap_size: int = 19,
        base_resolution: int = 14,
        max_resolution: int = 4096,
        use_viewdirs: bool = True,
        c2f: typing.Optional[float] = None,
        num_input_dim: int = 3,
        device: str = 'cpu',
        testing: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_frame = num_frame
        self.use_viewdirs = use_viewdirs
        self.nerf = NGPRadianceField(aabb=aabb, 
                                     num_dim=num_input_dim,
                                     num_layers=num_layers,
                                     hidden_dim=hidden_dim,
                                     geo_feat_dim=geo_feat_dim,
                                     n_levels=n_levels,
                                     log2_hashmap_size=log2_hashmap_size,
                                     base_resolution=base_resolution,
                                     n_features_per_level=n_features_per_level,
                                     max_resolution=max_resolution,                              
                                     use_viewdirs=use_viewdirs)
        self.c2f = c2f

        # # Set degrees of freedom
        # assert adjustment_type in ["none", "full", "rotation", "translation"]
        # self.adjustment_type = adjustment_type
        # dof = 3
        # if self.adjustment_type == "full":
        #     dof = 6

        # # noise addition for blender.
        # noise = torch.randn(num_frame, 6, device=device)*torch.tile(torch.tensor(noise), [num_frame, 1]).to(device)
        # self.pose_noise = se3_to_SE3(noise) # [1, 3, 4]

        # # Learnable embedding. 
        # self.pose_parameters = torch.nn.Embedding(num_frame, dof, device=device)
        # torch.nn.init.zeros_(self.pose_parameters.weight)

        # TODO:see if we want to register buffer for this variable.
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.k = torch.arange(n_levels-1, dtype=torch.float32, device=device)
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.device = device
        self.testing = testing

    def query_density(self, x, occ_sample=False):
        if self.c2f is None or (self.testing and not occ_sample):
            return self.nerf.query_density(x)
        return self.nerf.query_density(x, weights=self.get_weights())


    def get_weights(self):
        c2f_start, c2f_end = self.c2f
        # to prevent all weights to be 0, we leave the features from the first grid alone.
        alpha = (self.progress.data - c2f_start) / (c2f_end - c2f_start) * (self.n_levels-1)
        weight = (1 - (alpha - self.k).clamp_(min=0., max=1.).mul_(np.pi).cos_()) / 2
        weights = torch.cat([torch.ones(self.n_features_per_level, device=self.device), weight.repeat_interleave(self.n_features_per_level)])
        return weights


    def forward(self, positions, directions):
        if self.c2f is None or self.testing:
            return self.nerf(positions, directions)
        return self.nerf(positions, directions, weights=self.get_weights())

    def get_poses(self, idx=None, gt_poses=None, mode='train'):
        poses = gt_poses
        if mode == "train":
            pose_noises = self.pose_noise[idx]
            noisey_poses = compose_poses([pose_noises, gt_poses])
            if self.adjustment_type != "none":
                # add learnable pose correction
                assert idx is not None, "idx cannot be None during training."
                pose_parameters = self.pose_parameters.weight[idx] # [B, 6] idx corresponds to frame number.
                if self.adjustment_type == "full":
                    pose_adjustment = se3_to_SE3(pose_parameters) # [1, 3, 4]
                elif self.adjustment_type == "rotation":
                    pose_adjustment = so3_to_SE3(pose_parameters) # [1, 3, 4]
                else:
                    pose_adjustment = t3_to_SE3(pose_parameters) # [1, 3, 4]
                # add learnable pose correction
                poses = compose_poses([pose_adjustment, noisey_poses])
            else:
                poses = noisey_poses
        return poses

    # def query_rays(self, grid_3D, idx=None, gt_poses=None, mode='train'):
    #     """
    #     Assumption: perspective camera.

    #     Args:
    #         grid_3D: 3D grid in camera coordinate (training: pre-sampled).
    #         idx: frame ids.
    #         gt_poses: ground truth world-to-camera poses.
    #         mode: "train", "val", "test", "eval", "test-optim"
    #     """
    #     poses = self.get_poses(idx=idx, gt_poses=gt_poses, mode=mode)
    #     # # given the intrinsic/extrinsic matrices, get the camera center and ray directions
    #     center_3D = torch.zeros_like(grid_3D) # [B, N, 3]
    #     # transform from camera to world coordinates
    #     grid_3D = cam2world(grid_3D, poses) # [B, N, 3], [B, 3, 4] -> [B, 3]
    #     center_3D = cam2world(center_3D, poses) # [B, N, 3]
    #     directions = grid_3D - center_3D # [B, N, 3]
    #     viewdirs = directions / torch.linalg.norm(
    #         directions, dim=-1, keepdims=True
    #     )
    #     if mode in ['train', 'test-optim']:
    #         # This makes loss calculate easier. No more reshaping is needed.
    #         center_3D = torch.reshape(center_3D, (-1, 3))
    #         viewdirs = torch.reshape(viewdirs, (-1, 3))
    #     return Rays(origins=center_3D, viewdirs=viewdirs)
    
    def query_rays(self, photo_pts, camera_models, idx=None, mode='train'):
        """
        Args:
            photo_pts: [num_pts, 2] or [rows, cols, 2] row/column pixel coordinates
            camera_models: [num_images] list of camera models for each frame
            idx: [num_pts] or [1] camera ids
            mode: "train", "val", "test", "eval", "test-optim"
        """
        assert len(idx) == 1 | len(idx) == len(photo_pts)
        assert all(0 <= i <= len(camera_models) for i in idx)
        origins = []
        dirs = []
        for pt_idx, pt in enumerate(photo_pts):
            cam_idx = idx[0] if len(idx) == 1 else idx[pt_idx]
            camera_model = camera_models[cam_idx]
            origin, dir = camera_model.project_from_camera(pt)
            origins.append(origin)
            dirs.append(dir)

        # poses = self.get_poses(idx=idx, gt_poses=gt_poses, mode=mode)
        # # # given the intrinsic/extrinsic matrices, get the camera center and ray directions
        # center_3D = torch.zeros_like(grid_3D) # [B, N, 3]
        # # transform from camera to world coordinates
        # grid_3D = cam2world(grid_3D, poses) # [B, N, 3], [B, 3, 4] -> [B, 3]
        # center_3D = cam2world(center_3D, poses) # [B, N, 3]
        # directions = grid_3D - center_3D # [B, N, 3]
        # viewdirs = directions / torch.linalg.norm(
        #     directions, dim=-1, keepdims=True
        # )
        if mode in ['train', 'test-optim']:
            # This makes loss calculate easier. No more reshaping is needed.
            center_3D = torch.reshape(center_3D, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
        return Rays(origins=center_3D, viewdirs=viewdirs)

    def update_progress(self, progress):
        self.progress.data.fill_(progress)

