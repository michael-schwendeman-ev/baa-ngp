"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import imageio.v2 as imageio
import json
import numpy as np
import os
from PIL import Image
import sys
import torch
from utils import Rays
import torch.nn.functional as F
sys.path.append("..")

from camera_utils import img2cam, cam2world
from lie_utils import se3_to_SE3
from pose_utils import to_hom, construct_pose, compose_poses, invert_pose
import datasets.data_utils as du


def parse_raw_camera(pose_raw):
    """Convert pose from camera_to_world to world_to_camera and follow the right, down, forward coordinate convention."""
    pose_flip = construct_pose(R=torch.diag(torch.tensor([1,-1,-1]))) # right, up, backward --> right down, forward
    pose = compose_poses([pose_flip, pose_raw[:3]])
    pose = invert_pose(pose) # world_from_camera --> camera_from_world
    return pose
    

class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "black",
        num_rays: int = None,
        factor: float = 1,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        dof: int = 6,
        noise: float = 0.15
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.factor = factor
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.noise = noise
        self.images, self.depths, self.c2w, self.K, self.aabb, self.num_train_img = \
            du.load_renderings(root_fp, subject_id, 0, split)

        self.images = torch.from_numpy(self.images).to(torch.uint8).to(device)
        self.depths = torch.from_numpy(self.depths).to(torch.float32).to(device)
        self.height, self.width = self.images.shape[1:3]
        self.camfromworld = torch.from_numpy(self.c2w[:, :3, :]).to(torch.float32).to(device)
        self.K = torch.from_numpy(self.K).to(torch.float32).to(device)
        self.aabb = torch.tensor(self.aabb, dtype=torch.float32, device=device)

        self.no_ba = False
        self.OPENGL_CAMERA = True
        self.num_rays = num_rays
        self.training = (self.num_rays is not None) and (
            split in ["train", "trainval"]
        )

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba = data["rgba"]
        def rgba_to_pixels(rgba):
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

            if self.training:
                if self.color_bkgd_aug == "random":
                    color_bkgd = torch.rand(3, device=self.images.device)
                elif self.color_bkgd_aug == "white":
                    color_bkgd = torch.ones(3, device=self.images.device)
                elif self.color_bkgd_aug == "black":
                    color_bkgd = torch.zeros(3, device=self.images.device)
            else:
                # just use white during inference
                color_bkgd = torch.ones(3, device=self.images.device)

            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
            return pixels, color_bkgd
        pixels, color_bkgd = rgba_to_pixels(rgba)
        if self.training:
            return {
                "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
                "color_bkgd": color_bkgd,  # [3,]
                **{k: v for k, v in data.items() if k not in ["rgba"]},
            }

        # pre-calculate camera centers in camera coordinate to 
        # save time during training.
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # Compute image coordinate grid.
        if self.training:
            # Randomly select num_ray images and sample one ray per image.
            # allow duplicates, so one image may have more rays.
            if self.batch_over_images:
                assert self.num_rays is not None, "self.training is True, must pass in a num_rays."
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(self.num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.width, size=(self.num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(self.num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
        # adds 0.5 here.
        xy_grid = torch.stack([x,y],dim=-1).view(-1, 2) + 0.5 # [HW,2] or [B, N_rays, 2]
        # self.K is of shape [3,3]
        #grid_3D = img2cam(to_hom(xy_grid)[:, None, :], self.K[image_id]) # [B, 1, 2], [B, 3, 3] -> [B, 1, 3]
        images = self.images
        rgba = images[image_id, y, x] / 255.0

        w2c = torch.reshape(self.camfromworld[image_id], (-1, 3, 4)) # [3, 4] or (num_rays, 3, 4)
        
        K = self.K[image_id]
        grid_3D = F.pad(
            torch.stack(
                [
                    (x - K[:,0, 2] + 0.5) / K[:,0, 0],
                    (y - K[:,1, 2] + 0.5) / K[:,1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        if self.training:
            rgba = torch.reshape(rgba, (-1, 4))
            #grid_3D = torch.reshape(grid_3D, (-1, 1, 3)) # extra dimension is needed for query_rays.
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id, # [num_images]]
            }
        else:
            rgba = torch.reshape(rgba, (self.height, self.width, 4))
            grid_3D = torch.reshape(grid_3D, (self.height, self.width, 3))
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id, # [num_images]]
            }


def get_rays(x, y, K, c2w, OPENGL_CAMERA=True):
    import torch.nn.functional as F
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[:,0, 2] + 0.5) / K[:,0, 0],
                (y - K[:,1, 2] + 0.5) / K[:,1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]

    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    return origins, directions, camera_dirs