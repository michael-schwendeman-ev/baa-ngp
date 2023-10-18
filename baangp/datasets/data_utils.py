import os
import boto3
import json
import imageio.v2 as imageio
import numpy as np
from io import BytesIO
from scipy.spatial.transform import Rotation as SR
from typing import List

import cv2


def getROICornerPixels(c2w,fl,cx,cy,w,Pw):
    """Calculates the projected points of ROI corner points (Pw) in image space defined by intrinsics and c2w

    Args:
        c2w (_type_): camera to world matrix for image
        fl (_type_): focal length of the camera in pixels
        cx (_type_): principal point x
        cy (_type_): principal point y
        w (_type_): width of the image
        Pw (_type_): 3D world points to project into 2D space 

    Returns:
        pts: 2D pixel coordinates of Pw in specific order to be used in opencv masking
    """
    #intrinsic camera parameters of the image
    K = np.array([ [ fl,    0.000,   cx,      0.000], 
                    [ 0.000, fl,      cy,      0.000], 
                    [ 0.000, 0.000,   1.000,   0.000], 
                    [ 0.000, 0.000,   0.000,   1.000]])
                    
    Pe = np.linalg.inv(c2w) @ Pw #transform tile corners to camera space

    #image projection
    p = K @ Pe
    p[0] = w - p[0]/p[2]
    p[1] = p[1]/p[2]
    pts = p[0:2,[0,1,3,2]].astype(int)
    pts = pts.T.reshape(1,4,2)
    return pts

def maskImage(image, pts):
    """Masks image by keeping only pixels in a polygon defined by pts. All other pixels converted into transparent.
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    ignore_mask_color = (255,255,255)
    cv2.fillPoly(mask, pts, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    masked_image = np.stack((masked_image[:,:,0],masked_image[:,:,1],masked_image[:,:,2],mask[:,:,0]),axis=2)

    alive_fraction = np.count_nonzero(mask[...,0]) / mask[...,0].size #find ratio of non-empty pixels
    return masked_image, alive_fraction
