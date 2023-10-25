# Base class for all camera models
# \author Mike Schwendeman
# \date 23 Oct 2023


import sys
import json
import os
import typing
import torch
import numpy as np

sys.path.append("..")

from lie_utils import se3_to_SE3, so3_to_SE3, t3_to_SE3
from pose_utils import to_hom, construct_pose, compose_poses, invert_pose
from camera_utils import img2cam, cam2world
from utils import Rays

def parse_raw_camera(pose_raw):
    """Convert pose from camera_to_world to world_to_camera and follow the right, down, forward coordinate convention."""
    pose_flip = construct_pose(R=torch.diag(torch.tensor([1,-1,-1]))) # right, up, backward --> right down, forward
    pose = compose_poses([pose_flip, pose_raw[:3]])
    return pose

def load_camera_models(root_fp: str, subject_id: str, crop_borders: int = 0, split: str = 'train'):
    data_dir = os.path.join(root_fp, subject_id)
    meta = json.load(open(os.path.join(data_dir, f'transforms_{split}.json'), 'r'))
    camtoworld = []
    intrinsics = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
        cx = cx - crop_borders if crop_borders else cx
        cy = cy - crop_borders if crop_borders else cy
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)
        camtoworld.append(parse_raw_camera(torch.tensor(frame["transform_matrix"], dtype=torch.float32)))
    
    intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).to(torch.float32)
    camtoworld = torch.stack(camtoworld, axis=0)
    return camtoworld, intrinsics

class CameraModel(torch.nn.Module):
    """ Generates rays from camera models, given image ids and pixel coordinates
    """

    def __init__(
        self,         
        subject_id: str,
        root_fp: str,
        split: str,
        device: str = 'cpu',
        adjustment_type: str = "full",
        noise: typing.List = [0., 0., 0., 0., 0., 0.]
    ) -> None:
        super().__init__()
        camtoworld, intrinsics = load_camera_models(root_fp, subject_id, 0, split)
        num_frame = camtoworld.shape[0]
        self.camtoworld = camtoworld.to(device)
        self.intrinsics = intrinsics.to(device)
        
        # Set degrees of freedom
        assert adjustment_type in ["none", "full", "rotation", "translation"]
        self.adjustment_type = adjustment_type
        dof = 3
        if self.adjustment_type == "full":
            dof = 6

        # noise addition for blender.
        noise = torch.randn(num_frame, 6, device=device)*torch.tile(torch.tensor(noise), [num_frame, 1]).to(device)
        self.pose_noise = se3_to_SE3(noise) # [1, 3, 4]

        # Learnable embedding. 
        self.pose_parameters = torch.nn.Embedding(num_frame, dof, device=device)
        torch.nn.init.zeros_(self.pose_parameters.weight)

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

    def query_rays(self, grid_3D, idx=None, gt_poses=None, mode='train'):
        """
        Assumption: perspective camera.

        Args:
            grid_3D: 3D grid in camera coordinate (training: pre-sampled).
            idx: frame ids.
            gt_poses: ground truth world-to-camera poses.
            mode: "train", "val", "test", "eval", "test-optim"
        """
        poses = self.get_poses(idx=idx, gt_poses=gt_poses, mode=mode)
        # # given the intrinsic/extrinsic matrices, get the camera center and ray directions
        center_3D = torch.zeros_like(grid_3D) # [B, N, 3]
        # transform from camera to world coordinates
        grid_3D = cam2world(grid_3D, poses) # [B, N, 3], [B, 3, 4] -> [B, 3]
        center_3D = cam2world(center_3D, poses) # [B, N, 3]
        directions = grid_3D - center_3D # [B, N, 3]
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        if mode in ['train', 'test-optim']:
            # This makes loss calculate easier. No more reshaping is needed.
            center_3D = torch.reshape(center_3D, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
        return Rays(origins=center_3D, viewdirs=viewdirs)

    def forward(self, image_id, xy_grid, mode='train'):
        h, w, dims = xy_grid.size()
        assert dims == 2
        xy_flat = xy_grid.reshape(-1, 2)
        intrinsics = self.intrinsics[image_id]
        grid_3D = img2cam(to_hom(xy_flat)[:, None, :], intrinsics) # [B, 1, 2], [B, 3, 3] -> [B, 1, 3]
        grid_3D = grid_3D.reshape(h, w, 3)
        camtoworld = torch.reshape(self.camtoworld[image_id], (-1, 3, 4)) # [3, 4] or (num_rays, 3, 4)
        return self.query_rays(grid_3D, image_id, camtoworld, mode)
