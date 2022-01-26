# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np

from typing import List

from .detect import BoundingBox


def cast_pn_to_xyz(p_dst, normal, cam_v):
    """
    Cast plane-distance + normal inputs into camera xyz coordinate space

    Args:
        p_dst: a float list with shape [..., 1], recording the plane-distance from camera position
        normal: a float list with shape [..., 3], recording the normal direction related to camera space
        cam_v: a float list with shape [..., 3], recording the eye direction of each elements

    Returns:
        a float list with shape [..., 3], the camera space coordinate
    """
    normal_norm = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    cam_v_norm = cam_v / np.linalg.norm(cam_v, axis=-1, keepdims=True)
    cosine = np.sum(normal_norm * cam_v_norm, axis=-1, keepdims=True)
    cosine = np.where(cosine == 0., 0.000001, cosine)
    fov_xyz = p_dst / cosine * cam_v_norm
    return fov_xyz


def cast_d_to_xyz(d_dst: np.ndarray, cam_v):
    # cam_v_norm = cam_v / cam_v[..., :1]
    if d_dst.ndim == 2:
        d_dst = np.expand_dims(d_dst, axis=-1)
    fov_xyz = d_dst * cam_v
    return fov_xyz


def mask_vox_by_bbox(vox_size: List, bbox: BoundingBox):
    dim_index = [np.arange(a, dtype=np.int32) for a in vox_size]
    dim_index = np.stack(np.meshgrid(*dim_index, indexing='ij'), axis=-1)
    pos_mask = np.ones(vox_size, dtype=np.uint8)
    neg_mask = np.zeros(vox_size, dtype=np.uint8)
    dim_mask_lower = np.where(np.all(dim_index >= bbox.min, axis=-1), pos_mask, neg_mask)
    dim_mask_upper = np.where(np.all(dim_index <= bbox.max, axis=-1), pos_mask, neg_mask)
    dim_mask = np.where(np.logical_and(dim_mask_upper, dim_mask_lower), pos_mask, neg_mask).astype(np.int32)
    return dim_mask


def map_vox_into_coord(vox_size, vox_stride=1):
    dim_index = [np.arange(a, dtype=np.int32) for a in vox_size]
    dim_index = np.stack(np.meshgrid(*dim_index, indexing='ij'), axis=-1).astype(np.float32)
    dim_index = dim_index * vox_stride + vox_stride / 2
    return dim_index
